#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
report_engine.py — محرك التقرير المؤسسي الشامل

يُولّد تقريراً متكاملاً يشمل:
  • نسبة Sharpe و Sortino
  • أقصى خسائر متتالية
  • سلاسة منحنى الأسهم (Equity Curve Smoothness)
  • درجة الاستقرار (Stability Score)
  • معدل الفوز لكل نظام سوقي
  • تصدير JSON هيكلي

يُستدعى دورياً من main_bot أو عند الطلب من المشرف.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np

from config import config
from performance_tracker import TradeRecord, PerformanceTracker

logger = logging.getLogger(__name__)

# ─── ثوابت ───────────────────────────────────────────────────────────────────
TRADING_DAYS_YEAR = 252
RISK_FREE_DAILY   = config.RISK_FREE_RATE / TRADING_DAYS_YEAR


@dataclass
class InstitutionalReport:
    """التقرير المؤسسي الكامل — كل حقل موثق."""
    generated_at: str

    # ── مقاييس أساسية ─────────────────────────────────────────────────────────
    total_trades: int        = 0
    win_rate: float          = 0.0    # %
    profit_factor: float     = 0.0
    avg_rr: float            = 0.0    # R-multiple
    expectancy_r: float      = 0.0

    # ── مقاييس نسبة المخاطر ───────────────────────────────────────────────────
    sharpe_ratio: float      = 0.0    # عائد زائد / انحراف معياري
    sortino_ratio: float     = 0.0    # مثل Sharpe لكن يعاقب التقلب السلبي فقط
    calmar_ratio: float      = 0.0    # العائد السنوي / أقصى سحب

    # ── مقاييس السحب ──────────────────────────────────────────────────────────
    max_drawdown: float      = 0.0    # %
    avg_drawdown: float      = 0.0    # %
    recovery_factor: float   = 0.0   # إجمالي العائد / أقصى سحب

    # ── سلاسل الخسارة ─────────────────────────────────────────────────────────
    max_consec_losses: int   = 0
    max_consec_wins: int     = 0
    current_streak: int      = 0      # موجب = فوز، سالب = خسارة

    # ── جودة منحنى الأسهم ────────────────────────────────────────────────────
    equity_smoothness: float = 0.0   # 0 = متعرج، 1 = مثالي
    stability_score: float   = 0.0   # درجة إجمالية 0–100

    # ── أداء لكل نظام سوقي ───────────────────────────────────────────────────
    regime_performance: Dict[str, Dict] = field(default_factory=dict)

    # ── آخر 50 صفقة ──────────────────────────────────────────────────────────
    rolling_win_rate: float  = 0.0
    rolling_expectancy: float = 0.0

    # ── تقييم شامل ───────────────────────────────────────────────────────────
    system_grade: str        = "N/A"  # A+ → F
    alerts: List[str]        = field(default_factory=list)


class ReportEngine:
    """يُولّد InstitutionalReport من سجل الصفقات."""

    def __init__(self, tracker: PerformanceTracker) -> None:
        self._tracker = tracker

    def generate(self) -> InstitutionalReport:
        records = [r for r in self._tracker._records if r.result in ("win", "loss")]

        report = InstitutionalReport(
            generated_at=datetime.now(timezone.utc).isoformat()
        )

        if len(records) < 5:
            report.alerts.append("بيانات غير كافية لحساب المقاييس المؤسسية (< 5 صفقات)")
            return report

        # سلسلة R-multiples
        r_series = self._build_r_series(records)
        equity   = self._build_equity_curve(r_series)

        report.total_trades  = len(records)
        report.win_rate      = self._win_rate(records)
        report.profit_factor = self._profit_factor(records)
        report.avg_rr        = float(np.mean(r_series))
        report.expectancy_r  = self._expectancy(records)

        report.sharpe_ratio  = self._sharpe(r_series)
        report.sortino_ratio = self._sortino(r_series)
        report.max_drawdown  = self._max_drawdown(equity)
        report.avg_drawdown  = self._avg_drawdown(equity)
        report.calmar_ratio  = self._calmar(r_series, report.max_drawdown)
        report.recovery_factor = (
            (equity[-1] - 1.0) / report.max_drawdown
            if report.max_drawdown > 0 else float("inf")
        )

        mc_wins, mc_losses, cur = self._streak_stats(records)
        report.max_consec_losses = mc_losses
        report.max_consec_wins   = mc_wins
        report.current_streak    = cur

        report.equity_smoothness = self._smoothness(equity)
        report.stability_score   = self._stability_score(report)

        report.regime_performance = self._by_regime(records)

        rolling = records[-config.ROLLING_WINDOW:]
        r_rolling = self._build_r_series(rolling)
        report.rolling_win_rate    = self._win_rate(rolling)
        report.rolling_expectancy  = float(np.mean(r_rolling)) if r_rolling.size else 0.0

        report.system_grade = self._grade(report)
        report.alerts.extend(self._generate_alerts(report))

        self._save(report)
        return report

    # ── حسابات المقاييس ───────────────────────────────────────────────────────

    @staticmethod
    def _build_r_series(records: List[TradeRecord]) -> np.ndarray:
        series = []
        for r in records:
            if r.pnl_r is not None:
                series.append(r.pnl_r)
            else:
                series.append(config.DEFAULT_RR_RATIO if r.result == "win" else -1.0)
        return np.array(series, dtype=float)

    @staticmethod
    def _build_equity_curve(r_series: np.ndarray, risk: float = 0.01) -> np.ndarray:
        equity = [1.0]
        for r in r_series:
            equity.append(equity[-1] * (1 + risk * r))
        return np.array(equity)

    @staticmethod
    def _win_rate(records: List[TradeRecord]) -> float:
        if not records:
            return 0.0
        return sum(1 for r in records if r.result == "win") / len(records) * 100

    @staticmethod
    def _profit_factor(records: List[TradeRecord]) -> float:
        wins   = sum(r.pnl_r for r in records if r.result == "win"   and r.pnl_r)
        losses = sum(abs(r.pnl_r) for r in records if r.result == "loss" and r.pnl_r)
        return wins / losses if losses > 0 else float("inf")

    @staticmethod
    def _expectancy(records: List[TradeRecord]) -> float:
        r_vals = [
            r.pnl_r if r.pnl_r is not None else
            (config.DEFAULT_RR_RATIO if r.result == "win" else -1.0)
            for r in records
        ]
        return float(np.mean(r_vals)) if r_vals else 0.0

    @staticmethod
    def _sharpe(r_series: np.ndarray) -> float:
        if len(r_series) < 2:
            return 0.0
        excess = r_series - RISK_FREE_DAILY
        std = np.std(excess, ddof=1)
        if std == 0:
            return 0.0
        return float(np.mean(excess) / std * np.sqrt(TRADING_DAYS_YEAR))

    @staticmethod
    def _sortino(r_series: np.ndarray) -> float:
        if len(r_series) < 2:
            return 0.0
        excess    = r_series - RISK_FREE_DAILY
        downside  = excess[excess < 0]
        if len(downside) < 2:
            return float(np.mean(excess)) * TRADING_DAYS_YEAR
        downside_std = np.std(downside, ddof=1)
        if downside_std == 0:
            return 0.0
        return float(np.mean(excess) / downside_std * np.sqrt(TRADING_DAYS_YEAR))

    @staticmethod
    def _max_drawdown(equity: np.ndarray) -> float:
        peak = np.maximum.accumulate(equity)
        dd   = (peak - equity) / peak
        return float(np.max(dd))

    @staticmethod
    def _avg_drawdown(equity: np.ndarray) -> float:
        peak = np.maximum.accumulate(equity)
        dd   = (peak - equity) / peak
        in_drawdown = dd[dd > 0]
        return float(np.mean(in_drawdown)) if len(in_drawdown) else 0.0

    @staticmethod
    def _calmar(r_series: np.ndarray, max_dd: float) -> float:
        if max_dd == 0 or len(r_series) == 0:
            return 0.0
        annual_return = float(np.mean(r_series)) * TRADING_DAYS_YEAR * 0.01
        return annual_return / max_dd

    @staticmethod
    def _streak_stats(records: List[TradeRecord]) -> Tuple[int, int, int]:
        """يُعيد (أقصى فوز متتالي، أقصى خسارة متتالية، السلسلة الحالية)."""
        max_w = max_l = cur = 0
        streak_w = streak_l = 0
        for r in records:
            if r.result == "win":
                streak_w += 1; streak_l = 0
                cur = streak_w
            else:
                streak_l += 1; streak_w = 0
                cur = -streak_l
            max_w = max(max_w, streak_w)
            max_l = max(max_l, streak_l)
        return max_w, max_l, cur

    @staticmethod
    def _smoothness(equity: np.ndarray) -> float:
        """
        سلاسة منحنى الأسهم.
        نقارن منحنى الأسهم الفعلي بخط مستقيم مثالي.
        0 = متعرج تماماً، 1 = خط مستقيم تماماً.
        """
        if len(equity) < 3:
            return 0.0
        ideal   = np.linspace(equity[0], equity[-1], len(equity))
        max_dev = np.abs(equity - ideal).max()
        range_  = max(equity) - min(equity)
        if range_ == 0:
            return 1.0
        return float(max(0.0, 1.0 - max_dev / range_))

    @staticmethod
    def _stability_score(r: InstitutionalReport) -> float:
        """
        درجة إجمالية من 100 بناءً على مجموعة مقاييس.
        """
        score = 0.0

        # Sharpe (وزن 25)
        score += min(25, max(0, r.sharpe_ratio * 10))

        # Sortino (وزن 20)
        score += min(20, max(0, r.sortino_ratio * 8))

        # معدل الفوز (وزن 20)
        score += min(20, max(0, (r.win_rate - 40) * 0.8))

        # سلاسة منحنى الأسهم (وزن 15)
        score += r.equity_smoothness * 15

        # خطر السحب (وزن 20 — عقوبة)
        drawdown_penalty = min(20, r.max_drawdown * 100)
        score -= drawdown_penalty

        return round(max(0, min(100, score)), 1)

    @staticmethod
    def _by_regime(records: List[TradeRecord]) -> Dict[str, Dict]:
        """تجميع الأداء حسب نوع النظام السوقي المُسجَّل."""
        regimes: Dict[str, List] = {}
        for r in records:
            regime = getattr(r, "regime", "unknown") or "unknown"
            if regime not in regimes:
                regimes[regime] = []
            regimes[regime].append(r)

        result = {}
        for regime, recs in regimes.items():
            wins = sum(1 for r in recs if r.result == "win")
            result[regime] = {
                "trades":   len(recs),
                "win_rate": round(wins / len(recs) * 100, 1) if recs else 0,
            }
        return result

    @staticmethod
    def _grade(r: InstitutionalReport) -> str:
        """درجة النظام الإجمالية."""
        s = r.stability_score
        if s >= 80:  return "A+"
        if s >= 70:  return "A"
        if s >= 60:  return "B+"
        if s >= 50:  return "B"
        if s >= 40:  return "C"
        if s >= 30:  return "D"
        return "F"

    @staticmethod
    def _generate_alerts(r: InstitutionalReport) -> List[str]:
        alerts = []
        if r.sharpe_ratio < 1.0:
            alerts.append(f"⚠️ Sharpe منخفض ({r.sharpe_ratio:.2f}) — مخاطر عالية نسبياً")
        if r.max_drawdown > 0.20:
            alerts.append(f"🚨 سحب أقصى خطير ({r.max_drawdown:.1%})")
        if r.max_consec_losses >= config.KILL_SWITCH_CONSECUTIVE_LOSSES:
            alerts.append(f"⚠️ أقصى خسائر متتالية: {r.max_consec_losses}")
        if r.rolling_win_rate < 45:
            alerts.append(f"📉 معدل فوز متدحرج منخفض: {r.rolling_win_rate:.1f}%")
        if r.equity_smoothness < 0.5:
            alerts.append("📉 منحنى الأسهم متعرج — أداء غير مستقر")
        return alerts

    # ── تصدير ─────────────────────────────────────────────────────────────────

    def _save(self, report: InstitutionalReport) -> None:
        try:
            with open(config.REPORT_FILE, "w", encoding="utf-8") as f:
                json.dump(asdict(report), f, indent=2, ensure_ascii=False)
            logger.info("تم حفظ التقرير المؤسسي: %s", config.REPORT_FILE)
        except OSError as e:
            logger.error("فشل حفظ التقرير: %s", e)

    def format_telegram(self, report: Optional[InstitutionalReport] = None) -> str:
        if report is None:
            report = self.generate()
        alerts_str = "\n".join(report.alerts) if report.alerts else "لا تنبيهات"
        return (
            f"🏦 *التقرير المؤسسي*\n"
            f"━━━━━━━━━━━━━━\n"
            f"الدرجة: *{report.system_grade}* | الاستقرار: {report.stability_score}/100\n"
            f"إجمالي الصفقات: {report.total_trades}\n"
            f"معدل الفوز: {report.win_rate:.1f}%  (متدحرج: {report.rolling_win_rate:.1f}%)\n"
            f"عامل الربح: {report.profit_factor:.2f}\n"
            f"التوقع: {report.expectancy_r:+.4f} R\n"
            f"━━━━━━━━━━━━━━\n"
            f"Sharpe: {report.sharpe_ratio:.2f} | Sortino: {report.sortino_ratio:.2f}\n"
            f"أقصى سحب: {report.max_drawdown:.1%}\n"
            f"معامل التعافي: {report.recovery_factor:.2f}\n"
            f"سلاسة المنحنى: {report.equity_smoothness:.2f}\n"
            f"━━━━━━━━━━━━━━\n"
            f"أقصى خسائر متتالية: {report.max_consec_losses}\n"
            f"السلسلة الحالية: {report.current_streak:+d}\n"
            f"━━━━━━━━━━━━━━\n"
            f"*التنبيهات:*\n{alerts_str}\n"
        )
