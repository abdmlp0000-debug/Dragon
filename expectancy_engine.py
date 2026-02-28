#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
expectancy_engine.py — محرك التوقع الرياضي وخطر الإفلاس

يحسب ويتتبع:
  • التوقع الرياضي (Expectancy) لكل صفقة بوحدات R
  • خطر الإفلاس (Risk of Ruin) بناءً على احتمالات الفوز/الخسارة
  • كسر كيلي المعدَّل (Fractional Kelly) لتحديد حجم المركز الأمثل
  • محاكاة تآكل رأس المال (Capital Decay) في سيناريوهات سلبية
  • التوقع المتدحرج على آخر N صفقة

يُوقف النظام الإشارات إذا تدهور التوقع أو ارتفع خطر الإفلاس.
"""

from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Tuple

import numpy as np

from config import config

logger = logging.getLogger(__name__)


@dataclass
class ExpectancySnapshot:
    """لحظة تقييم لمقاييس التوقع والمخاطر."""
    win_rate: float          # احتمال الفوز (0–1)
    rr_ratio: float          # نسبة المكافأة للمخاطرة
    expectancy_r: float      # التوقع بوحدات R
    risk_of_ruin: float      # احتمال الإفلاس (0–1)
    kelly_fraction: float    # كسر كيلي المثالي (بدون تعديل)
    safe_kelly: float        # كسر كيلي الآمن (مُعدَّل × 0.25)
    rolling_expectancy: float  # التوقع المتدحرج على آخر N
    capital_decay_50: float   # تآكل رأس المال عند 50 صفقة خاسرة متتالية
    signals_allowed: bool    # هل يجب إيقاف الإشارات؟
    reason: str              # سبب الإيقاف إن وُجد


class ExpectancyEngine:
    """
    يتتبع نتائج الصفقات ويُصدر ExpectancySnapshot عند كل طلب.
    يُستدعى من RiskManager ليُقرر السماح بالإشارات أو إيقافها.
    """

    def __init__(self) -> None:
        self._history: Deque[Tuple[str, float]] = deque(maxlen=200)
        # كل إدخال: ("win" | "loss", pnl_r)
        self._rolling: Deque[Tuple[str, float]] = deque(maxlen=config.ROLLING_WINDOW)

    # ── واجهة عامة ────────────────────────────────────────────────────────────

    def record(self, result: str, pnl_r: float = None) -> None:
        """تسجيل نتيجة صفقة مكتملة."""
        if pnl_r is None:
            pnl_r = config.DEFAULT_RR_RATIO if result == "win" else -1.0
        self._history.append((result, pnl_r))
        self._rolling.append((result, pnl_r))

    def evaluate(self) -> ExpectancySnapshot:
        """يُحسب الصورة الكاملة لمقاييس التوقع والمخاطر."""
        win_prob, rr = self._estimate_params()
        exp_r        = self._expectancy(win_prob, rr)
        ror          = self._risk_of_ruin(win_prob, rr)
        kelly        = self._kelly(win_prob, rr)
        safe_k       = kelly * config.KELLY_FRACTION
        rolling_exp  = self._rolling_expectancy()
        decay_50     = self._capital_decay(win_prob, rr, n=50)
        allowed, reason = self._gate(exp_r, ror, rolling_exp)

        snap = ExpectancySnapshot(
            win_rate=win_prob,
            rr_ratio=rr,
            expectancy_r=round(exp_r, 4),
            risk_of_ruin=round(ror, 4),
            kelly_fraction=round(kelly, 4),
            safe_kelly=round(safe_k, 4),
            rolling_expectancy=round(rolling_exp, 4),
            capital_decay_50=round(decay_50, 4),
            signals_allowed=allowed,
            reason=reason,
        )

        if not allowed:
            logger.warning("ExpectancyEngine: إيقاف الإشارات — %s", reason)
        return snap

    # ── الحسابات الإحصائية ───────────────────────────────────────────────────

    def _estimate_params(self) -> Tuple[float, float]:
        """يستنتج win_prob و rr من التاريخ المتاح أو يستخدم القيم الافتراضية."""
        if len(self._history) < 10:
            return config.DEFAULT_WIN_PROB, config.DEFAULT_RR_RATIO

        results = list(self._history)
        wins   = [(r, p) for r, p in results if r == "win"]
        losses = [(r, p) for r, p in results if r == "loss"]

        win_prob = len(wins) / len(results)
        avg_win  = float(np.mean([p for _, p in wins]))   if wins   else config.DEFAULT_RR_RATIO
        avg_loss = float(np.mean([abs(p) for _, p in losses])) if losses else 1.0
        rr = avg_win / avg_loss if avg_loss > 0 else config.DEFAULT_RR_RATIO

        return win_prob, rr

    @staticmethod
    def _expectancy(win_prob: float, rr: float) -> float:
        """
        E = (P_win × RR) − (P_loss × 1.0)
        قيمة موجبة = حافة إيجابية
        """
        return (win_prob * rr) - ((1 - win_prob) * 1.0)

    @staticmethod
    def _risk_of_ruin(win_prob: float, rr: float, n_units: int = 20) -> float:
        """
        صيغة خطر الإفلاس للمراهنة الثنائية المتكاملة:
        RoR = ((1 - E) / (1 + E))^n_units
        حيث E = الحافة الموحدة (بين 0 و1)
        """
        exp = (win_prob * rr) - ((1 - win_prob) * 1.0)
        if exp <= 0:
            return 1.0   # حافة سلبية → إفلاس مؤكد
        edge = exp / (win_prob * rr + (1 - win_prob))  # تطبيع
        edge = min(max(edge, 0.0), 0.9999)
        ror  = ((1 - edge) / (1 + edge)) ** n_units
        return float(ror)

    @staticmethod
    def _kelly(win_prob: float, rr: float) -> float:
        """
        صيغة كيلي: K = (P × RR − Q) / RR
        K < 0 → لا تتداول
        """
        q = 1 - win_prob
        k = (win_prob * rr - q) / rr
        return max(0.0, k)

    def _rolling_expectancy(self) -> float:
        """التوقع على آخر N صفقة."""
        if len(self._rolling) < 5:
            return config.DEFAULT_WIN_PROB * config.DEFAULT_RR_RATIO - (1 - config.DEFAULT_WIN_PROB)
        results = list(self._rolling)
        wins  = [(r, p) for r, p in results if r == "win"]
        total = len(results)
        if total == 0:
            return 0.0
        wp = len(wins) / total
        avg_win  = float(np.mean([p for _, p in wins])) if wins else config.DEFAULT_RR_RATIO
        losses   = [(r, p) for r, p in results if r == "loss"]
        avg_loss = float(np.mean([abs(p) for _, p in losses])) if losses else 1.0
        rr = avg_win / avg_loss if avg_loss > 0 else config.DEFAULT_RR_RATIO
        return (wp * rr) - ((1 - wp) * 1.0)

    @staticmethod
    def _capital_decay(win_prob: float, rr: float, n: int = 50) -> float:
        """
        تآكل رأس المال المتوقع بعد N صفقة متتالية خاسرة
        (سيناريو أسوأ الحالات).
        يُستخدم كمقياس استقرار.
        """
        loss_prob = 1 - win_prob
        # احتمال n خسائر متتالية
        prob_streak = loss_prob ** n
        # تأثير كل خسارة على رأس المال (نسبة مخاطرة = 1% لكل صفقة)
        capital_after = (1 - 0.01) ** n
        return float(capital_after * prob_streak)  # القيمة المتوقعة للتآكل

    # ── بوابة الإشارات ────────────────────────────────────────────────────────

    @staticmethod
    def _gate(exp_r: float, ror: float, rolling_exp: float) -> Tuple[bool, str]:
        """
        يُقرر السماح بالإشارات أو وقفها بناءً على مقاييس موضوعية.
        """
        if ror >= config.MAX_RISK_OF_RUIN:
            return False, f"خطر الإفلاس مرتفع جداً: {ror:.1%} ≥ {config.MAX_RISK_OF_RUIN:.1%}"

        if exp_r < config.EXPECTANCY_MIN:
            return False, f"توقع سلبي: {exp_r:.3f} < {config.EXPECTANCY_MIN}"

        if rolling_exp < config.EXPECTANCY_MIN * 0.5:
            return False, f"توقع متدحرج متدهور: {rolling_exp:.3f}"

        return True, "حافة إيجابية مقبولة"

    # ── تقرير نصي ────────────────────────────────────────────────────────────

    def format_report(self) -> str:
        s = self.evaluate()
        return (
            f"📐 *تقرير التوقع الرياضي*\n"
            f"━━━━━━━━━━━━━━\n"
            f"احتمال الفوز: {s.win_rate:.1%}\n"
            f"نسبة R/R: {s.rr_ratio:.2f}\n"
            f"التوقع (E): {s.expectancy_r:+.4f} R\n"
            f"التوقع المتدحرج: {s.rolling_expectancy:+.4f} R\n"
            f"خطر الإفلاس: {s.risk_of_ruin:.2%}\n"
            f"كسر كيلي الآمن: {s.safe_kelly:.1%}\n"
            f"الإشارات: {'✅ مسموحة' if s.signals_allowed else '🛑 موقوفة'}\n"
            f"السبب: {s.reason}\n"
        )
