#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
config.py — المركز الوحيد لجميع معاملات المحرك المؤسسي.
لا توجد أرقام سحرية داخل منطق العمل — كل قيمة قابلة للضبط هنا.
النسخة v3 — مستوى مؤسسي.
"""

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class Config:
    # ── Telegram ──────────────────────────────────────────────────────────────
    TOKEN: str     = "8491771221:AAHsUdmNWSy-_Ljp8s2ieP6nr_RbzGjBABw"
    ADMIN_ID: int  = 7853240017

    # ── الأزواج المتداولة ─────────────────────────────────────────────────────
    SYMBOLS: Dict[str, str] = field(default_factory=lambda: {
        "EUR/USD": "EURUSD=X",
        "GBP/USD": "GBPUSD=X",
        "AUD/USD": "AUDUSD=X",
        "USD/JPY": "JPY=X",
        "USD/CAD": "CAD=X",
        "USD/CHF": "CHF=X",
        "NZD/USD": "NZDUSD=X",
        "BTC/USD": "BTC-USD",
        "ETH/USD": "ETH-USD",
        "XAU/USD": "GC=F",
        "XAG/USD": "SI=F",
        "US30":    "YM=F",
        "GER40":   "DAX=F",
    })

    # ── الإطارات الزمنية ───────────────────────────────────────────────────────
    HIGHER_TF: str     = "5m"
    LOWER_TF: str      = "1m"
    HIGHER_PERIOD: str = "2d"
    LOWER_PERIOD: str  = "1d"

    # ── جودة البيانات ─────────────────────────────────────────────────────────
    MIN_DATA_POINTS: int         = 60
    OUTLIER_STD_THRESH: float    = 5.0
    MAX_GAP_MULTIPLIER: float    = 2.5
    EQUAL_LEVEL_TOLERANCE: float = 0.0003

    # ── هيكل السوق ────────────────────────────────────────────────────────────
    FRACTAL_WINDOW: int = 5

    # ── فلاتر الزخم ───────────────────────────────────────────────────────────
    ADX_MIN_TREND: int         = 25
    ADX_STRONG_TREND: int      = 30
    VOLUME_SPIKE_FACTOR: float = 1.5
    RSI_OVERSOLD: float        = 35.0
    RSI_OVERBOUGHT: float      = 65.0

    # ── عتبات الثقة الأساسية ──────────────────────────────────────────────────
    MIN_CONFIDENCE: float = 70.0

    # ── ضوابط المخاطر ────────────────────────────────────────────────────────
    MAX_SIGNALS_PER_15M: int            = 3
    COOLDOWN_AFTER_LOSS_MINUTES: int    = 3
    KILL_SWITCH_CONSECUTIVE_LOSSES: int = 3
    KILL_SWITCH_RESET_WINS: int         = 2

    # ── تتبع الأداء ───────────────────────────────────────────────────────────
    ROLLING_WINDOW: int = 50

    # ── [جديد] كشف النظام السوقي (Regime Detection) ─────────────────────────
    REGIME_ADX_TRENDING: float    = 28.0   # ADX فوق هذا → اتجاهي
    REGIME_ADX_RANGING: float     = 18.0   # ADX تحت هذا → عرضي
    REGIME_ATR_HV_MULT: float     = 1.8    # مضاعف ATR للتمييز بين تقلب عالٍ/منخفض
    REGIME_ATR_LV_MULT: float     = 0.6    # تقلب منخفض جداً → ضغط

    # عتبات الثقة لكل نظام سوقي
    REGIME_CONF_TRENDING: float   = 68.0
    REGIME_CONF_RANGING: float    = 78.0   # أشد صرامةً في الأسواق العرضية
    REGIME_CONF_HIGH_VOL: float   = 80.0   # أشد صرامةً في التقلب العالي
    REGIME_CONF_LOW_VOL: float    = 75.0

    # حد الإشارات لكل نظام (كل 15 دقيقة)
    REGIME_RATE_TRENDING: int     = 4
    REGIME_RATE_RANGING: int      = 1
    REGIME_RATE_HIGH_VOL: int     = 1
    REGIME_RATE_LOW_VOL: int      = 2

    # ── [جديد] Walk-Forward Validation ───────────────────────────────────────
    WFV_TRAIN_BARS: int    = 300   # شموع التدريب لكل نافذة
    WFV_TEST_BARS: int     = 100   # شموع الاختبار (خارج العينة)
    WFV_MIN_WINDOWS: int   = 3     # أدنى عدد نوافذ مقبول
    WFV_MIN_WIN_RATE: float = 0.45  # أدنى معدل فوز مقبول عبر النوافذ
    WFV_MAX_INSTABILITY: float = 0.25  # أقصى تباين مقبول بين النوافذ

    # ── [جديد] محرك Expectancy و Risk-of-Ruin ────────────────────────────────
    DEFAULT_WIN_PROB: float    = 0.55   # احتمال الفوز الافتراضي قبل جمع بيانات كافية
    DEFAULT_RR_RATIO: float    = 1.5    # نسبة مكافأة/مخاطرة افتراضية
    MAX_RISK_OF_RUIN: float    = 0.05   # 5% → إيقاف الإشارات إذا تجاوز
    KELLY_FRACTION: float      = 0.25   # كسر كيلي (25% من كيلي الكامل للتحفظ)
    EXPECTANCY_MIN: float      = 0.05   # حد أدنى للتوقع الإيجابي (بالـ R)

    # ── [جديد] محاكاة الانزلاق والسبريد ──────────────────────────────────────
    SLIPPAGE_MIN_PIPS: float   = 0.2    # أدنى انزلاق (نقطة أساس)
    SLIPPAGE_MAX_PIPS: float   = 1.5    # أقصى انزلاق عشوائي
    SPREAD_BASE_PIPS: float    = 0.5    # سبريد عادي
    SPREAD_VOL_MULT: float     = 2.5    # مضاعف السبريد في التقلب العالي
    EXEC_DELAY_MS_MIN: int     = 50     # تأخير تنفيذ أدنى (ملي ثانية)
    EXEC_DELAY_MS_MAX: int     = 300    # تأخير تنفيذ أقصى

    # ── [جديد] Monte Carlo ───────────────────────────────────────────────────
    MC_SIMULATIONS: int        = 1000   # عدد المحاكاة
    MC_MAX_DRAWDOWN_LIMIT: float = 0.30  # 30% → النظام هش
    MC_STABILITY_MIN: float    = 0.60   # مؤشر استقرار أدنى

    # ── [جديد] حماية رأس المال ───────────────────────────────────────────────
    CAPITAL_DRAWDOWN_THROTTLE: float  = 0.10   # 10% سحب → خفض الإشارات
    CAPITAL_DRAWDOWN_HALT: float      = 0.20   # 20% سحب → وقف كامل
    CAPITAL_DRAWDOWN_COOLDOWN_H: int  = 4      # ساعات انتظار إجباري بعد الوقف

    # ── [جديد] Sharpe/Sortino parameters ────────────────────────────────────
    RISK_FREE_RATE: float      = 0.04   # معدل خالٍ من المخاطرة (سنوي)

    # ── الملفات ───────────────────────────────────────────────────────────────
    SUBSCRIBERS_FILE: str  = "subscribers.json"
    SIGNALS_LOG_FILE: str  = "signals_log.jsonl"
    PERFORMANCE_FILE: str  = "performance.json"
    REPORT_FILE: str       = "institutional_report.json"


# نقطة وصول وحيدة للمشروع كله
config = Config()

