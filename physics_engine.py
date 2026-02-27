# physics_engine.py
# محرك فيزياء بسيط لمحاكاة التحليق والتأثيرات البيئية (رياح، جاذبية، ريش، مجهود)
# بدون مكتبات خارجية ثقيلة – فقط math

import math
from typing import Any, Dict, List, Optional, Union
from typing import Dict, Union   # ← أضف ده في أعلى الملف إذا مش موجود

from ai_npl_e import generate_wind_effect_description
from ai_npl_e import AiNplE
try:
    from ai_npl_e import AiNplE
except ImportError as e:
    print(f"خطأ في استيراد ai_npl_e: {e}")
    AiNplE = None  # fallback مؤقت
    
# ────────────────────────────────────────────────
# ثوابت فيزيائية عامة
# ────────────────────────────────────────────────

GRAVITY = 9.81                  # m/s²
AIR_DENSITY = 1.225             # kg/m³
DEFAULT_DRAG_COEFF = 0.35       # معامل سحب تقريبي لنسر
DEFAULT_LIFT_COEFF = 1.6        # معامل رفع تقريبي لنسر
MAX_ANGLE_OF_ATTACK = 15.0      # درجة (فوقها stall محتمل)
STALL_ANGLE_THRESHOLD = 18.0    # درجة (بدء الاضطراب)


# ────────────────────────────────────────────────
# دوال مساعدة رياضية أساسية
# ────────────────────────────────────────────────

def drag_force(v_rel: float, area: float, cd: float = DEFAULT_DRAG_COEFF) -> float:
    """قوة السحب (Drag)"""
    return 0.5 * AIR_DENSITY * (v_rel ** 2) * cd * area

def lift_force(v_rel: float, area: float, cl: float = DEFAULT_LIFT_COEFF) -> float:
    """قوة الرفع (Lift)"""
    return 0.5 * AIR_DENSITY * (v_rel ** 2) * cl * area

# ────────────────────────────────────────────────
# 1. دالة حساب زاوية الهجوم الديناميكية (Active Control)
# ────────────────────────────────────────────────

def adjust_angle_of_attack(
    current_lift_ratio: float,
    wind_speed_ms: float,
    current_aoa_deg: float = 10.0,
    max_aoa: float = MAX_ANGLE_OF_ATTACK,
    min_aoa: float = 4.0
) -> float:
    """
    النسر يعدل زاوية الهجوم تلقائيًا للحفاظ على التوازن أو تقليل المجهود
    """
    adjustment_factor = 1.0

    if current_lift_ratio > 1.12:
        adjustment_factor = 0.82   # قلل الزاوية عشان ما يرتفع كتير
    elif current_lift_ratio < 0.92:
        adjustment_factor = 1.18   # زد الزاوية عشان يحافظ على الارتفاع
    elif current_lift_ratio < 0.98:
        adjustment_factor = 1.08

    new_aoa = current_aoa_deg * adjustment_factor

    # تأثير الرياح القوية (تقليل الزاوية المطلوبة)
    if wind_speed_ms > 12:
        new_aoa *= 0.88

    return max(min_aoa, min(new_aoa, max_aoa))

def generate_wind_effect_description(wind_speed_kmh: float) -> str:
    """
    توليد وصف نصي دقيق لتأثير الرياح حسب سرعتها (بالكم/ساعة)
    مناسب للإضافة إلى prompt البيئي في AI.NPL(E)
    """
    if wind_speed_kmh <= 1:
        return "calm, almost no wind, gentle air movement, barely noticeable breeze"

    elif 1 < wind_speed_kmh <= 10:
        return "light breeze, soft wind 5-10 km/h, leaves rustling quietly, subtle hair and fur movement"

    elif 10 < wind_speed_kmh <= 20:
        return "moderate breeze, steady wind 10-20 km/h, leaves and small branches swaying, gentle rippling on fabric and fur"

    elif 20 < wind_speed_kmh <= 38:
        return "fresh wind 20-38 km/h, strong breeze, branches moving noticeably, dust and small debris swirling lightly, hair and clothing fluttering"

    elif 38 < wind_speed_kmh <= 50:
        return "strong wind 38-50 km/h, near gale force, trees swaying heavily, loose objects rattling, dust clouds rising, significant force on feathers, fur, and clothing"

    elif 50 < wind_speed_kmh <= 75:
        return "high wind 50-75 km/h, gale force, large branches breaking, walking difficult, dust storms forming, strong pressure on wings and surfaces, objects blown away"

    elif 75 < wind_speed_kmh <= 118:
        return "storm-force wind 75-118 km/h, violent storm, trees uprooted, roofs damaged, very strong turbulence, extreme force on all exposed surfaces and creatures"

    elif 118 < wind_speed_kmh <= 200:
        return "hurricane-force wind 118-200 km/h, destructive hurricane, massive structural damage, flying debris dangerous, extreme turbulence and pressure, near-impossible flight for birds"

    else:
        return "catastrophic wind exceeding 200 km/h, violent tornado/hurricane, total devastation, airborne objects lethal, impossible stability, chaotic turbulent flow"
       
def calculate_soaring_physics(
        bird_mass: float,
        wing_area: float,
        wind_speed_kmh: float,
        bird_forward_speed_kmh: float = 40,
        feather_flex_coeff: float = 0.6,
        dt: float = 0.033
    ) -> dict:
        """
        حساب فيزياء التحليق في مواجهة الرياح
        """
        wind_speed = wind_speed_kmh / 3.6
        bird_speed = bird_forward_speed_kmh / 3.6

        airspeed = bird_speed + wind_speed

        weight_force = bird_mass * GRAVITY

        lift = lift_force(airspeed, wing_area)

        drag = drag_force(airspeed, wing_area * 0.4)

        lift_vs_weight_ratio = lift / weight_force

        feather_flex = feather_flex_coeff * (airspeed / 20)

        effort_level = max(0.1, 1.0 - (wind_speed / 15))

        return {
            "airspeed_ms": round(airspeed, 2),
            "lift_force_N": round(lift, 2),
            "weight_force_N": round(weight_force, 2),
            "lift_ratio": round(lift_vs_weight_ratio, 3),
            "drag_force_N": round(drag, 2),
            "feather_flex_factor": round(feather_flex, 2),
            "effort_level": round(effort_level, 2),
            "status": (
                "stable soaring" if 0.95 <= lift_vs_weight_ratio <= 1.05 else
                "gaining altitude" if lift_vs_weight_ratio > 1.05 else
                "losing altitude"
            )
        }
          
@staticmethod
def wind_effect_on_object(
    wind_speed_kmh: float,           # سرعة الرياح بالكم/س
    object_mass: float,              # كتلة الجسم (كجم)
    object_area: float,              # مساحة المقطع العرضي (m²)
    object_cd: float = DEFAULT_DRAG_COEFF,  # معامل السحب
    lift_coeff: float = 0.0,         # معامل الرفع (0 للأجسام غير الرفيعة)
    gravity_on: bool = True,         # هل نطبق الجاذبية؟
    dt: float = 0.1                  # خطوة زمنية (ثواني)
) -> dict:
    """
    محاكاة تأثير الرياح + الجاذبية على جسم لخطوة زمنية واحدة
    ترجع: القوى، التسارع، التغير في السرعة
    """
    # تحويل سرعة الرياح إلى m/s
    wind_speed = wind_speed_kmh / 3.6

    # قوة السحب (في اتجاه الرياح)
    drag = drag_force(wind_speed, object_area, object_cd)

    # قوة الرفع (عمودي على الرياح تقريباً)
    lift = lift_force(wind_speed, object_area, lift_coeff)

    # قوة الجاذبية (إذا مفعلة)
    gravity_force = object_mass * GRAVITY if gravity_on else 0.0

    # القوى الصافية (نفترض الرياح أفقية، الجاذبية رأسية)
    net_force_x = drag   # الرياح تدفع أفقياً
    net_force_y = lift - gravity_force  # الرفع لأعلى، الجاذبية لأسفل

    # التسارع (F = m a → a = F/m)
    ax = net_force_x / object_mass
    ay = net_force_y / object_mass

    # تغير السرعة خلال dt
    dvx = ax * dt
    dvy = ay * dt

    return {
        "wind_speed_ms": wind_speed,
        "drag_force": round(drag, 2),
        "lift_force": round(lift, 2),
        "gravity_force": round(gravity_force, 2),
        "net_force_x": round(net_force_x, 2),
        "net_force_y": round(net_force_y, 2),
        "acceleration_x": round(ax, 2),
        "acceleration_y": round(ay, 2),
        "velocity_change_x": round(dvx, 2),
        "velocity_change_y": round(dvy, 2),
    }
             
# ────────────────────────────────────────────────
# 2. دالة توزيع الضغط على الجناح/الريش (Pressure Distribution)
# ────────────────────────────────────────────────
def compute_feather_flex_distribution(
    airspeed_ms: float,
    wing_length_m: float = 1.0,
    max_flex: float = 2.0
) -> Dict[str, Union[float, str]]:
    # مثال: كلما كان الجناح أطول → الثني في الأطراف أكبر نسبيًا
    flex_tip = min(max_flex, airspeed_ms / 12.0 * (wing_length_m / 1.0))
    flex_mid = flex_tip * 0.65
    flex_root = flex_tip * 0.35

    return {
        "tip_flex": round(flex_tip, 2),
        "mid_flex": round(flex_mid, 2),
        "root_flex": round(flex_root, 2),
        "description": f"wingtip feathers flexed {flex_tip:.2f} (wing length {wing_length_m}m), mid-wing {flex_mid:.2f}, root {flex_root:.2f}"
    }
    
# ────────────────────────────────────────────────
# 3. دالة حساب المجهود العضلي الفعلي (Muscle Effort Model)
# ────────────────────────────────────────────────
def calculate_muscle_power(
    lift_force: float,
    drag_force: float,
    velocity_ms: float,
    efficiency: float = 0.25
) -> Dict[str, Union[float, str]]:   # ← أفضل حل: يقبل float أو str
    """
    تقدير القدرة/الطاقة المبذولة (Power = Force × Velocity)
    """
    total_force = math.sqrt(lift_force**2 + drag_force**2)
    power_required = total_force * velocity_ms / efficiency

    max_power_estimate = 120.0
    relative_effort = power_required / (max_power_estimate * 4.5)

    return {
        "power_required_w": round(power_required, 1),
        "relative_effort": round(relative_effort, 2),
        "description": f"muscle power required: {power_required:.1f} W, effort level {relative_effort*100:.0f}%"
    }

# ────────────────────────────────────────────────
# 4. دالة كشف الاضطراب / Stall
# ────────────────────────────────────────────────
def detect_turbulence_or_stall(
    airspeed_ms: float,
    aoa_deg: float,
    wind_gust_factor: float = 1.0
) -> Dict[str, Any]:  # ← Any بأحرف كبيرة
    stall_risk = False
    turbulence_risk = False
    warnings = []

    if aoa_deg > 17.0:
        stall_risk = True
        warnings.append("angle of attack too high → stall risk")

    if airspeed_ms < 8.0:
        stall_risk = True
        warnings.append("airspeed too low → stall risk")

    if wind_gust_factor > 1.3 or airspeed_ms > 25:
        turbulence_risk = True
        warnings.append("strong turbulence from high wind/gusts")

    return {
        "stall_risk": stall_risk,
        "turbulence_risk": turbulence_risk,
        "warnings": warnings,
        "description": "chaotic feather flutter and wing instability" if turbulence_risk or stall_risk else "stable flight"
    }

# ────────────────────────────────────────────────
# 5. دالة التكيف مع أنواع الكائنات (Adaptation Layer)
# ────────────────────────────────────────────────
def get_physics_params_for_entity(entity_type: str) -> Dict:
    """
    إرجاع معلمات فيزيائية افتراضية حسب نوع الكائن
    """
    params = {
        "cat": {"mass": 4.0, "area": 0.05, "lift_coeff": 0.0, "drag_coeff": 0.8},
        "eagle": {"mass": 4.5, "area": 1.1, "lift_coeff": 1.6, "drag_coeff": 0.35},
        "car": {"mass": 1500, "area": 2.5, "lift_coeff": 0.1, "drag_coeff": 0.3},
        "tree": {"mass": 500, "area": 10.0, "lift_coeff": 0.0, "drag_coeff": 1.2},
    }
    return params.get(entity_type.lower(), {"mass": 1.0, "area": 0.5, "lift_coeff": 0.5, "drag_coeff": 0.5})

# ────────────────────────────────────────────────
# 6. دالة ترجمة الفيزياء إلى نص prompt (Physics-to-Prompt Translator)
# ────────────────────────────────────────────────
def physics_to_prompt_text(physics: Dict, monitor: Dict, entity: str = "eagle") -> str:
    if AiNplE is None:
        wind_desc = "unknown wind effect (ai_npl_e not loaded)"
    else:
        wind_kmh = monitor.get("wind_kmh", 0) if monitor else 0
        wind_desc = AiNplE.generate_wind_effect_description(wind_kmh)

    lines = []
    if monitor.get("status") == "stable soaring":
        lines.append(f"the {entity} is soaring steadily against the wind")
    elif monitor.get("status") == "gaining altitude":
        lines.append(f"the {entity} is rising smoothly with the help of strong headwind")
    else:
        lines.append(f"the {entity} is struggling slightly against the wind")

    effort = physics.get("effort_level", monitor.get("effort_estimate", 1.0))
    if effort < 0.5:
        lines.append("minimal wing flapping effort thanks to the assisting wind")

    flex = physics.get("feather_flex_factor", monitor.get("feather_flex", 0.0))
    if flex > 0.8:
        lines.append(f"feathers visibly flexing and ruffling in the {wind_speed_kmh} km/h wind")

    warnings = monitor.get("warnings", [])
    if warnings:
        lines.append(f"slight visual stress on feathers due to wind pressure: {', '.join(warnings)}")

    return f"{entity} soaring in {wind_desc}, dramatic atmosphere, ultra realistic, " + ", ".join(lines)

# ────────────────────────────────────────────────
# 7. دالة محاكاة متعددة الخطوات (Multi-step Simulation)
# ────────────────────────────────────────────────
def simulate_over_time(
    entity_params: Dict,
    wind_speed_kmh: float,
    duration_sec: float = 1.0,
    fps: int = 30,
    entity_name: str = "eagle"
) -> List[Dict]:
    dt = 1.0 / fps
    num_steps = int(duration_sec * fps)

    core = PhysicsCore()
    instance = EntityPhysicsInstance(core, entity_name, entity_params)
    monitor = PhysicsMonitor()

    frames = []

    for step in range(num_steps):
        step_result = instance.step(wind_speed_kmh, dt)
        check = monitor.check(instance, step_result)

        prompt_text = physics_to_prompt_text(
            step_result["forces"],
            check,
            entity_name,
            wind_speed_kmh
        )

        frames.append({
            "frame": step,
            "time": round(step * dt, 2),
            "physics": step_result,
            "monitor": check,
            "prompt_snippet": prompt_text
        })

    return frames

# ────────────────────────────────────────────────
# المحرك الفيزيائي الرئيسي (مشترك)
# ────────────────────────────────────────────────
class PhysicsCore:
    
    @staticmethod
    def drag_force(v: float, area: float, cd: float = DEFAULT_DRAG_COEFF) -> float:
        return 0.5 * AIR_DENSITY * (v ** 2) * cd * area

    @staticmethod
    def lift_force(v: float, area: float, cl: float = DEFAULT_LIFT_COEFF) -> float:
        """
        قوة الرفع (Lift force) = ½ × ρ × v² × Cl × A
        (Cl = معامل الرفع، يعتمد على زاوية الهجوم)
        """
        return 0.5 * AIR_DENSITY * (v ** 2) * cl * area

    @staticmethod
    def compute_forces(
        self,
        mass: float,
        area: float,
        airspeed: float,
        lift_coeff: float = 1.2,   # قللناه ليكون أقرب للواقع
        drag_coeff: float = DEFAULT_DRAG_COEFF,
    ) -> Dict[str, float]:
        weight = mass * GRAVITY
        lift = PhysicsCore.lift_force(airspeed, area, lift_coeff)
        drag = PhysicsCore.drag_force(airspeed, area * 0.4, drag_coeff)  # مساحة سحب أصغر

        return {
            "weight_N": round(weight, 2),
            "lift_N": round(lift, 2),
            "drag_N": round(drag, 2),
            "lift_ratio": round(lift / weight, 3) if weight > 0 else 999.0,
            "airspeed_ms": round(airspeed, 2)
        }

    def magnitude(self, vx: float, vy: float) -> float:
        """حساب مقدار المتجه (السرعة أو القوة)"""
        return math.sqrt(vx**2 + vy**2)

    def compute_soaring_effort(
        self,                        # ← أضف هذا السطر فقط (مهم جدًا)
        bird_mass_kg: float = 2.2,
        wing_area_m2: float = 1.1,
        lift_coeff: float = 1.5,
        wind_speed_kmh: float = 20.0,
        min_airspeed_for_lift_kmh: float = 38.0
    ) -> dict:
        """
        حساب مجهود التحليق مع رياح مقابلة
        """
        # الوزن بالنيوتن
        weight_N = bird_mass_kg * 9.81

        # السرعة النسبية المطلوبة بدون رياح (لحساب أدنى Lift = Weight)
        v_min_ms = math.sqrt( (2 * weight_N) / (AIR_DENSITY * wing_area_m2 * lift_coeff) )
        v_min_kmh = v_min_ms * 3.6

        # سرعة الرياح بالمتر/ث
        wind_ms = wind_speed_kmh / 3.6

        # السرعة النسبية الفعلية إذا طار النسر بسرعته الطبيعية
        effective_airspeed_ms = v_min_ms + wind_ms
        effective_airspeed_kmh = effective_airspeed_ms * 3.6

        # نسبة المجهود (كلما زادت الرياح → قلّ المجهود)
        effort_ratio = max(0.1, v_min_ms / effective_airspeed_ms)  # 1.0 = مجهود كامل، 0.1 = مجهود خفيف

        # هل يستطيع التحليق بثبات؟
        can_soar = effective_airspeed_kmh >= v_min_kmh

        return {
            "weight_N": round(weight_N, 1),
            "min_airspeed_no_wind_kmh": round(v_min_kmh, 1),
            "wind_speed_kmh": wind_speed_kmh,
            "effective_airspeed_kmh": round(effective_airspeed_kmh, 1),
            "effort_ratio": round(effort_ratio, 2),
            "can_maintain_stable_soaring": can_soar,
            "description": (
                f"النسر يحلق بمجهود خفيف جدًا ({effort_ratio*100:.0f}%) بفضل الرياح المقابلة {wind_speed_kmh} كم/س"
                if can_soar and effort_ratio < 0.5 else
                "الرياح تساعد لكن المجهود لا يزال ملحوظًا"
            )
        }

    def active_control_adjustment(
        self,                        # ← أضف هذا السطر فقط (مهم جدًا)
        current_lift_ratio: float,
        wind_speed_ms: float,
        max_aoa_deg: float = 15.0
    ) -> float:
        """
        النسر يعدل زاوية الهجوم (angle of attack) تلقائيًا
        """
        if current_lift_ratio > 1.1:
            # الرفع زايد → قلل الزاوية عشان ما يرتفع كتير
            return max_aoa_deg * 0.7
        elif current_lift_ratio < 0.95:
            # الرفع ناقص → زد الزاوية عشان يحافظ على الارتفاع
            return max_aoa_deg * 1.15
        else:
            # مستقر → يحافظ على زاوية متوسطة
            return max_aoa_deg * 0.95
        
        # ────────────────────────────────────────────────
        # دالة Monitoring للتوافق بين NPLs
        # ────────────────────────────────────────────────

    def monitor_wind_effect_compatibility(self, physics_result: dict) -> dict:
        """
        تراقب توافق تأثير الرياح مع قدرة الكائن
        """
        ratio = physics_result["lift_ratio"]
        flex = physics_result["feather_flex_factor"]
        effort = physics_result["effort_level"]

        status = "optimal"
        warnings = []
        recommendations = []

        if ratio < 0.9:
            status = "critical"
            warnings.append("قوة الرفع غير كافية – النسر قد يهبط أو يفقد الارتفاع")
            recommendations.append("قلل سرعة الرياح أو زد مساحة الجناحين في الوصف")
        elif ratio > 1.15:
            status = "excessive_lift"
            warnings.append("الرفع قوي جداً – النسر قد يصعد بسرعة غير مرغوبة")
            recommendations.append("قلل معامل الرفع أو زاوية الجناحين")

        if flex > 1.2:
            status = "high_stress" if status == "optimal" else status
            warnings.append("ثني الريش كبير – قد يظهر اهتزاز أو تشوه بصري")
            recommendations.append("قلل سرعة الرياح أو أضف وصف 'ريش قوي'")

        if effort < 0.4:
            recommendations.append("الرياح تساعد كثيراً – مجهود خفيف جداً، تحليق سلس")

        return {
            "status": status,
            "lift_ratio": ratio,
            "feather_flex": flex,
            "effort_level": effort,
            "warnings": warnings,
            "recommendations": recommendations
        }

    def physics_to_prompt_text(
        self,                      # ← أضف هذا (ضروري)
        physics: Dict,             # نتيجة الحسابات الفيزيائية
        monitor: Dict,             # نتيجة المراقبة/التوافق
        entity: str = "eagle"
    ) -> str:
        """
        ترجمة النتائج الفيزيائية إلى وصف نصي جاهز للـ prompt
        """
        # وصف الرياح من AiNplE
        wind_kmh = monitor.get("wind_kmh", 0)  # افتراضي 0 إذا غير موجود
        wind_desc = AiNplE.generate_wind_effect_description(wind_kmh)

        lines = []

        # استخدام monitor بدل result (لأن result غير معرف)
        if monitor.get("status") == "stable soaring":
            lines.append(f"the {entity} is soaring steadily against the wind")
        elif monitor.get("status") == "gaining altitude":
            lines.append(f"the {entity} is rising smoothly with the help of strong headwind")
        else:
            lines.append(f"the {entity} is struggling slightly against the wind")

        # استخدام physics أو monitor حسب البيانات المتوفرة
        effort = physics.get("effort_level", monitor.get("effort_estimate", 1.0))
        if effort < 0.5:
            lines.append("minimal wing flapping effort thanks to the assisting wind")

        flex = physics.get("feather_flex_factor", monitor.get("feather_flex", 0.0))
        if flex > 0.8:
            lines.append(f"feathers visibly flexing and ruffling in the {wind_kmh} km/h wind")

        # التحذيرات من monitor
        warnings = monitor.get("warnings", [])
        if warnings:
            lines.append(f"slight visual stress on feathers due to wind pressure: {', '.join(warnings)}")

        # الوصف النهائي
        return f"{entity} soaring in {wind_desc}, dramatic atmosphere, ultra realistic, " + ", ".join(lines)
    
    # ────────────────────────────────────────────────
    # دالة Monitoring للتوافق بين NPLs
    # ────────────────────────────────────────────────

    @staticmethod
    def monitor_physics_compatibility(
        env_effect: dict,
        subject_response: dict,
    ) -> dict:
        """
        تراقب إذا كانت قوى البيئة متوافقة مع قدرة الكائن/الجسم
        """
        wind_speed = env_effect.get("wind_speed_ms", 0)
        drag = env_effect.get("drag_force", 0)
        lift = env_effect.get("lift_force", 0)

        mass = subject_response.get("mass", 1.0)
        max_lift = subject_response.get("max_lift_capacity", 0)

        status = "stable"
        warnings = []

        if wind_speed > 25:
            status = "high_wind_stress"
            warnings.append("قوة الرياح عالية – قد يحدث اهتزاز أو انحراف كبير")

        if lift < mass * GRAVITY * 0.8:
            status = "insufficient_lift"
            warnings.append("قوة الرفع غير كافية – قد يسقط الجسم أو يميل بشدة")

        if drag > subject_response.get("max_drag_resistance", drag * 2):
            status = "excessive_drag"
            warnings.append("قوة السحب كبيرة – قد يتحرك الجسم بسرعة غير مرغوبة")

        return {
            "status": status,
            "warnings": warnings,
            "recommendation": "apply wind effect normally" if not warnings else "reduce wind effect or increase resistance"
        }

# ────────────────────────────────────────────────
# نواة خاصة بكل كائن (طائر، نسر، سيارة...)
# ────────────────────────────────────────────────

class EntityPhysicsInstance:
    def __init__(self, core: PhysicsCore, name: str, params: Dict):
        self.core = core
        self.name = name
        self.params = params
        
        # تعريف موحد للحالة
        self.state = {
            "airspeed": 12.0,          # ← نفس الاسم في كل مكان
            "altitude": 300.0,
            "feather_flex": 0.0
        }

    def step(self, wind_speed_kmh: float, dt: float = 0.033) -> Dict:
        wind_ms = wind_speed_kmh / 3.6
        wind_description = AiNplE.generate_wind_effect_description(wind_speed_kmh)
        airspeed = self.state["airspeed"] + wind_ms
        
        # حساب النسبة المتوقعة
        predicted_lift = PhysicsCore.lift_force(airspeed, self.params["area"], self.params["lift_coeff"])
        predicted_ratio = predicted_lift / (self.params["mass"] * GRAVITY)
        
        # تعديل Cl (أكثر حساسية)
        adjusted_cl = self.adjust_lift_coeff(predicted_ratio, self.params["lift_coeff"])
        adjusted_cl = min(adjusted_cl, 1.0)  # حد أقصى واقعي
        
        forces = self.core.compute_forces(
            mass=self.params["mass"],
            area=self.params["area"],
            airspeed=airspeed,
            lift_coeff=adjusted_cl,
            drag_coeff=self.params.get("drag_coeff", DEFAULT_DRAG_COEFF)
        )
        
        self.state["airspeed"] = airspeed
        self.state["feather_flex"] = min(2.0, airspeed / 15.0)
        
        return {
            "forces": forces,
            "state": self.state.copy(),
            "airspeed_ms": round(airspeed, 2),
            "feather_flex": round(self.state["feather_flex"], 2),
            "wind_ms": round(wind_ms, 2),
            "wind_environmental_description": wind_description   # ← تحت إشراف ai_npl_e
        }
  
    def adjust_lift_coeff(self, current_ratio: float, base_cl: float) -> float:
        if current_ratio > 1.30:
            return base_cl * 0.50   # خفض قوي جدًا
        elif current_ratio > 1.15:
            return base_cl * 0.70
        elif current_ratio > 1.05:
            return base_cl * 0.85
        elif current_ratio < 0.90:
            return base_cl * 1.30
        elif current_ratio < 0.95:
            return base_cl * 1.15
        return base_cl
    
    def adjust_cl_for_stability(self, predicted_ratio: float, base_cl: float) -> float:
        """
        تعديل معامل الرفع تلقائيًا عشان نحافظ على استقرار قريب من 1.0
        """
        if predicted_ratio > 1.20:
            return base_cl * 0.70   # قلل الرفع بشكل ملحوظ
        elif predicted_ratio > 1.10:
            return base_cl * 0.85
        elif predicted_ratio < 0.90:
            return base_cl * 1.20
        elif predicted_ratio < 0.95:
            return base_cl * 1.10
        return base_cl
    
# ────────────────────────────────────────────────
# دالة Monitoring للتوافق والتحذيرات
# ────────────────────────────────────────────────

class PhysicsMonitor:
    """تراقب توافق الحسابات الفيزيائية مع قدرات الكائن"""

    def check(self, instance: EntityPhysicsInstance, physics_result: Dict) -> Dict:
        forces = physics_result["forces"]
        state = physics_result["state"]

        lift_ratio = forces["lift_ratio"]
        feather_flex = state["feather_flex"]
        airspeed = state["airspeed"]

        status = "stable"
        warnings = []
        recommendations = []

        # 1. الرفع مقابل الجاذبية
        if lift_ratio < 0.95:
            status = "insufficient_lift"
            warnings.append(f"قوة الرفع ({forces['lift']:.1f} N) غير كافية مقابل الوزن ({forces['weight']:.1f} N)")
            recommendations.append("زيادة مساحة الجناحين أو سرعة الطيران")
        elif lift_ratio > 1.15:
            status = "excessive_lift"
            warnings.append("الرفع قوي جدًا – قد يرتفع الكائن بسرعة غير مرغوبة")
            recommendations.append("تقليل زاوية الهجوم أو معامل الرفع")

        # 2. تأثير الرياح على الريش/الجناح
        if feather_flex > 1.2:
            status = "high_feather_stress" if status == "stable" else status
            warnings.append(f"ثني الريش/الجناح مرتفع (عامل {feather_flex:.2f}) بسبب السرعة النسبية {airspeed:.1f} m/s")
            recommendations.append("تقليل سرعة الرياح أو وصف ريش أقوى")

        # 3. المجهود العام
        effort = max(0.1, 1.0 - (airspeed / 30))  # تقريبي
        if effort < 0.4:
            recommendations.append("الرياح تساعد كثيراً – مجهود تحليق خفيف")

        return {
            "status": status,
            "lift_ratio": round(lift_ratio, 3),
            "feather_flex": round(feather_flex, 2),
            "effort_estimate": round(effort, 2),
            "warnings": warnings,
            "recommendations": recommendations
        }

# ────────────────────────────────────────────────
# دالة توليد خريطة الحلقات المتعرجة + نقاط خروج الريش
# ────────────────────────────────────────────────
def generate_body_cross_sections_and_feather_exits(
    total_length: float = 0.95,
    sections_count: int = 14,
    base_radius: float = 0.15,
    feather_density_per_meter: float = 7200,
    waviness_amplitude: float = 0.022,
    waviness_frequency: float = 19.0,
    avg_feather_length: float = 0.22,
    follicle_depth_mm: dict = None      # عمق البصيلة حسب المنطقة
) -> Dict:
    
    import math
    import random

    if follicle_depth_mm is None:
        follicle_depth_mm = {
            "neck": 2.5,
            "chest_back": 4.0,
            "wing_coverts": 5.0,
            "flight_feathers": 8.0,
            "tail": 10.0
        }

    sections = []
    all_feathers = []

    for i in range(sections_count):
        z = i / (sections_count - 1) * total_length

        # تحديد نوع المنطقة لاختيار العمق
        if z < 0.15: region = "neck"
        elif z < 0.45: region = "chest_back"
        elif z < 0.75: region = "wing_coverts"
        else: region = "tail"

        depth_m = follicle_depth_mm[region] / 1000  # تحويل مم إلى متر

        radius_factor = 0.38 + 0.95 * math.sin(math.pi * z / total_length)
        radius = base_radius * radius_factor

        angle_offset = waviness_amplitude * math.sin(waviness_frequency * math.pi * z / total_length)

        circumference_points = int(28 + 52 * radius_factor)
        density_adjust = feather_density_per_meter * (2 * math.pi * radius) / circumference_points

        section_feathers = []

        for j in range(circumference_points):
            if random.random() < density_adjust:
                angle = (j / circumference_points) * 360 + angle_offset * 180
                
                # نقطة البذرة (داخل الجسم)
                base_x = (radius - depth_m) * math.cos(math.radians(angle))
                base_y = (radius - depth_m) * math.sin(math.radians(angle))

                # نقطة الخروج (على السطح)
                exit_x = radius * math.cos(math.radians(angle))
                exit_y = radius * math.sin(math.radians(angle))

                feather = {
                    "id": len(all_feathers) + 1,
                    "region": region,
                    "base":  {"x": round(base_x, 4),  "y": round(base_y, 4),  "z": round(z, 4)},
                    "exit":  {"x": round(exit_x, 4),  "y": round(exit_y, 4),  "z": round(z, 4)},
                    "depth_mm": round(depth_m * 1000, 1),
                    "angle_deg": round(angle % 360, 1),
                    "projected_length": round(avg_feather_length, 3)
                }
                
                section_feathers.append(feather)
                all_feathers.append(feather)

        sections.append({
            "z": round(z, 4),
            "radius": round(radius, 3),
            "region": region,
            "feathers": section_feathers
        })

    return {
        "body_length_m": total_length,
        "section_count": sections_count,
        "total_feathers": len(all_feathers),
        "sections": sections,
        "all_feathers": all_feathers
    }
    
# ────────────────────────────────────────────────
# دالة محاكاة فيزيائية موزعة على نقاط الريش
# ────────────────────────────────────────────────
def compute_distributed_lift_from_exit_points(
    exit_points: List[Dict],
    airspeed_ms: float,
    wind_kmh: float = 0.0,
    base_cl: float = 1.35,
    aoa_deg: float = 8.0
) -> Dict:
    wind_ms = wind_kmh / 3.6
    effective_v = airspeed_ms + wind_ms

    total_lift = 0.0
    per_point_results = []

    for point in exit_points:
        area_per_feather = 0.012 + 0.008 * (1 - point["local_radius"] / 0.3)

        position_factor = 1.0 - 0.55 * (point["z"] ** 1.4)
        local_cl = base_cl * position_factor * math.sin(math.radians(aoa_deg * 1.15))

        lift_n = 0.5 * AIR_DENSITY * (effective_v ** 2) * area_per_feather * local_cl

        total_lift += lift_n

        per_point_results.append({
            "z": point["z"],
            "symbol": point["symbol"],
            "local_area_m2": round(area_per_feather, 5),
            "local_cl": round(local_cl, 3),
            "lift_N": round(lift_n, 3)
        })

    return {
        "total_lift_N": round(total_lift, 2),
        "effective_airspeed_ms": round(effective_v, 2),
        "point_count": len(exit_points),
        "average_lift_per_point_N": round(total_lift / len(exit_points), 3) if exit_points else 0,
        "per_point": per_point_results
    }
    
# ────────────────────────────────────────────────
# دالة رسم الجسم + خطوط الريش (base → exit) باستخدام matplotlib
# ────────────────────────────────────────────────
def plot_feather_map_2d(feather_map: Dict, view: str = "side", save_path: str = "feather_map.png"):
    """
    رسم بسيط ثنائي الأبعاد للنقاط والخطوط بين base و exit
    view: "side" (جانبي) أو "top" (من الأعلى)
    """
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_aspect('equal')
    ax.set_title(f"Feather Exit Map - {view} view")
    
    if view == "side":
        x_key, y_key = "x", "z"   # x أفقي، z طول الجسم
    else:  # top view
        x_key, y_key = "x", "y"
    
    # رسم الحلقات كدوائر تقريبية
    for sec in feather_map["sections"]:
        circle = plt.Circle((0, sec["z"]), sec["radius"], color='gray', fill=False, alpha=0.3, linestyle='--')
        ax.add_patch(circle)
    
    # رسم كل ريشة كخط صغير من base إلى exit + نقطة في المنتصف
    for feather in feather_map["all_feathers"]:
        bx, by = feather["base"][x_key], feather["base"][y_key]
        ex, ey = feather["exit"][x_key], feather["exit"][y_key]
        
        # خط بين base و exit
        ax.plot([bx, ex], [by, ey], color='black', linewidth=0.8, alpha=0.7)
        
        # نقطة في منتصف الريشة (لتمثيل الانحناء المستقبلي)
        mx = (bx + ex) / 2
        my = (by + ey) / 2
        ax.plot(mx, my, 'o', color='red', markersize=2, alpha=0.6)
    
    ax.set_xlabel(x_key)
    ax.set_ylabel(y_key)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"تم حفظ الصورة في: {save_path}")
    
# ────────────────────────────────────────────────
# دالة إضافة ثني/انحناء لكل ريشة بناءً على الـ vector
# ────────────────────────────────────────────────
def apply_feather_curvature(feather_map: Dict, wind_kmh: float = 20.0, base_curvature: float = 0.15) -> Dict:
    """
    إضافة انحناء لكل ريشة بناءً على vector (base → exit)
    يُرجع نسخة محدثة تحتوي على نقاط منحنية
    """
    import math
    import random

    wind_ms = wind_kmh / 3.6
    updated_feathers = []

    for feather in feather_map["all_feathers"]:
        bx, by, bz = feather["base"]["x"], feather["base"]["y"], feather["base"]["z"]
        ex, ey, ez = feather["exit"]["x"], feather["exit"]["y"], feather["exit"]["z"]

        # طول الـ vector
        dx, dy, dz = ex - bx, ey - by, ez - bz
        length = math.sqrt(dx**2 + dy**2 + dz**2)

        # شدة الانحناء (تزيد مع السرعة ومع طول الريشة)
        curvature_strength = base_curvature * (wind_ms / 10) * (length / 0.2)
        curvature_strength = min(curvature_strength, 0.45)  # حد أقصى للانحناء

        # اتجاه عمودي على الـ vector للانحناء (نختار اتجاه عشوائي معقول)
        perp_x = -dy if abs(dx) > abs(dy) else dz
        perp_y = dx
        perp_norm = math.sqrt(perp_x**2 + perp_y**2) + 1e-6
        perp_x /= perp_norm
        perp_y /= perp_norm

        # نقطة وسط منحنية
        mid_x = (bx + ex) / 2 + perp_x * curvature_strength * length
        mid_y = (by + ey) / 2 + perp_y * curvature_strength * length
        mid_z = (bz + ez) / 2

        feather["curved_mid"] = {
            "x": round(mid_x, 4),
            "y": round(mid_y, 4),
            "z": round(mid_z, 4)
        }

        updated_feathers.append(feather)

    feather_map["all_feathers"] = updated_feathers
    return feather_map
    
# ────────────────────────────────────────────────
# مثال استخدام كامل
# ────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== اختبار خريطة خروج الريش ===\n")
    
    # 1. توليد خريطة الحلقات + نقاط الريش
    feather_map = generate_body_cross_sections_and_feather_exits(
        total_length=0.95,
        sections_count=12,
        base_radius=0.15,
        feather_density_per_meter=8000,
        waviness_amplitude=0.02,
        waviness_frequency=18.0
    )
    
    # تطبيق الانحناء
    curved_map = apply_feather_curvature(feather_map, wind_kmh=25.0)
    
    # رسم النتيجة
    plot_feather_map_2d(curved_map, view="side", save_path="naser_feather_map_side.png")
    plot_feather_map_2d(curved_map, view="top", save_path="naser_feather_map_top.png")
    print(f"تم توليد {feather_map['total_feather_exits']} نقطة خروج ريش")
    print(f"عدد الحلقات: {feather_map['section_count']}")
    
    # عرض أول 3 حلقات كمثال
    print("\nأول 3 حلقات:")
    for sec in feather_map["sections"][:3]:
        print(f"  z={sec['z']:.3f} m | radius={sec['radius']:.3f} m | "
              f"نقاط خروج ريش: {len(sec['feather_exits'])}")
    
    print("\n=== محاكاة رفع موزع على نقاط الريش ===")
    
    # 2. محاكاة الرفع باستخدام النقاط المولدة
    physics_result = compute_distributed_lift_from_exit_points(
        exit_points=feather_map["exit_points"],
        airspeed_ms=14.0,          # سرعة أمامية تقريبية
        wind_kmh=20.0,
        base_cl=1.35,
        aoa_deg=8.0
    )
    
    print(f"الرفع الكلي: {physics_result['total_lift_N']} N")
    print(f"السرعة الفعالة: {physics_result['effective_airspeed_ms']} m/s")
    print(f"متوسط رفع لكل نقطة: {physics_result['average_lift_per_point_N']} N")
    print(f"عدد النقاط المحسوبة: {physics_result['point_count']}")