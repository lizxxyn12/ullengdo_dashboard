"""
CSS 스타일 정의 모듈

이 모듈은 대시보드에서 사용되는 모든 CSS 스타일을 포함합니다.
"""

# -----------------------------
# 글로벌 CSS 스타일
# -----------------------------
GLOBAL_CSS = """
<style>
/* 전체 폭 여백 조정 */
.block-container {
  padding-top: 2rem;
  padding-bottom: 2.4rem;
  max-width: 100%;
}

.notice-pill {
  width: 100%;
  margin-top: 0.5rem;
  line-height: 1.2;
  background: #f3f3f3;
  border-radius: 999px;
  padding: 14px 18px;
  font-weight: 400;
  color: #333;
  border: 1px solid #e6e6e6;
}

.dashboard-title {
  display: flex;
  align-items: center;
  gap: 12px;
  margin: 0.8rem 0 0.6rem 0;
}
.dashboard-title img {
  width: 40px;
  height: 40px;
  object-fit: contain;
}
.dashboard-title .title-text {
  font-size: 1.6rem;
  font-weight: 800;
  color: #1f1f1f;
}

.card-title {
  font-weight: 700;
  margin-bottom: 8px;
}
.card-sub {
  color: #666;
  font-size: 0.9rem;
}

.photo-placeholder {
  background: #e9f2ff;
  color: #0b5cab;
  border-radius: 16px;
  height: 250px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-weight: 400;
  font-size: 1.05rem;
}

.bus-detail {
  border: 1px solid #e8ebf2;
  background: #f8f9fc;
  border-radius: 14px;
  padding: 12px 14px;
  margin-bottom: 10px;
}
.bus-detail-title {
  font-weight: 800;
  font-size: 1.05rem;
  color: #1f1f1f;
}
.bus-detail-sub {
  color: #666;
  font-size: 0.85rem;
  margin-top: 4px;
}
.bus-route-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(170px, 1fr));
  gap: 8px;
}
.bus-route-card {
  border: 1px solid #e6e9f2;
  border-left: 5px solid #9aa3b2;
  border-radius: 12px;
  padding: 10px 10px 9px 10px;
  background: #ffffff;
}
.bus-route-id {
  font-weight: 700;
  color: #20232a;
}
.bus-route-desc {
  color: #555;
  font-size: 0.82rem;
  margin-top: 2px;
  line-height: 1.3;
}
.bus-route-empty {
  border: 1px dashed #d0d4de;
  color: #808899;
  border-radius: 12px;
  padding: 10px;
  background: #fafbfe;
  font-size: 0.9rem;
}

/* UI 요소 z-index 조정 */
div[data-baseweb="select"] { position: relative; z-index: 3000; }
div[data-baseweb="popover"] { z-index: 4000; }
section.main iframe { position: relative; z-index: 1; }
div[data-testid="stIFrame"] iframe { min-height: 360px; }

/* 다이얼로그 스타일 */
div[data-testid="stDialog"] > div { width: min(96vw, 1400px); margin: 0 auto; }
div[data-testid="stDialog"] div[role="dialog"] { max-height: 92vh; padding: 0; }
div[data-testid="stDialog"] img { max-height: 86vh; width: 100%; object-fit: contain; display: block; }

/* --- Card & Sea Notice Styles --- */
.r2-card {
  background: #f6f7fb;
  border: 1px solid #ebedf3;
  border-radius: 22px;
  padding: 18px 18px 16px 18px;
  box-sizing: border-box;
}
.r2-top {
  display: flex;
  flex-direction: column;
  gap: 4px;
}
.r2-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 12px;
}
.r2-title {
  font-size: 1.05rem;
  font-weight: 800;
}
.r2-date {
  color: #777;
  font-size: 0.85rem;
}
.r2-card-body {
  margin-top: 8px;
}

/* Sea Section & Layout */
.sea-section {
  background: #ffffff;
  border: 1px solid #e8ebf2;
  border-radius: 16px;
  padding: 12px;
  margin-bottom: 12px;
}
.sea-section-title {
  font-size: 0.82rem;
  font-weight: 500;
  color: #6b7280;
  margin-bottom: 8px;
  letter-spacing: 0.2px;
}
.sea-kpi-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 12px;
}
.sea-kpi-card {
  background: #ffffff;
  border: 1px solid #e8ebf2;
  border-radius: 14px;
  padding: 12px;
}
.sea-kpi-title {
  font-weight: 700;
  color: #1f2937;
}
.sea-kpi-value {
  font-size: 1.2rem;
  font-weight: 800;
  margin-top: 4px;
}
.sea-kpi-meta {
  color: #6b7280;
  font-size: 0.82rem;
  margin-top: 6px;
  line-height: 1.4;
}
.sea-badges {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}
.sea-badge {
  background: #fff4e5;
  color: #b54708;
  border: 1px solid #f4c790;
  border-radius: 999px;
  padding: 4px 10px;
  font-size: 0.78rem;
  font-weight: 700;
}
.sea-latest {
  display: flex;
  align-items: center;
  gap: 10px;
}
.sea-pill {
  background: #e8f0ff;
  color: #2f6bff;
  border-radius: 999px;
  padding: 6px 12px;
  font-weight: 700;
  font-size: 0.88rem;
}
.sea-latest-text {
  font-size: 1.02rem;
  font-weight: 700;
  color: #1d1d1d;
}

/* Bar Charts (Updated) */
.sea-bars {
  display: grid;
  gap: 12px;
  margin-bottom: 2px;
}
.bar-row {
  display: grid;
  grid-template-columns: 170px 1fr;
  gap: 10px;
  align-items: center;
}
.bar-label {
  font-weight: 600;
  font-size: 0.86rem;
}
.bar-label-wrap {
  display: flex;
  align-items: center;
  gap: 6px;
  flex-wrap: nowrap;
  white-space: nowrap;
}
.bar-sub {
  font-size: 0.82rem;
  font-weight: 400;
  color: #666;
}
.bar-track {
  background: #ffffff;
  border: 1px solid #edf0f5;
  border-radius: 999px;
  padding: 4px;
  position: relative;
}
.bar-fill {
  height: 14px;
  border-radius: 999px;
  position: relative;
}
.bar-fill-split {
  height: 14px;
  border-radius: 999px;
  overflow: hidden;
  display: flex;
  position: relative;
}
.bar-seg {
  height: 100%;
}
.bar-value-onfill {
  position: absolute;
  left: 50%;
  top: 50%;
  transform: translate(-50%, -50%);
  background: #ffffff;
  color: #374151;
  font-size: 0.74rem;
  font-weight: 700;
  padding: 2px 10px;
  border-radius: 999px;
  border: 1px solid rgba(0, 0, 0, 0.08);
  box-shadow: 0 4px 10px rgba(17, 24, 39, 0.1);
  pointer-events: none;
  white-space: nowrap;
}

/* Tooltip (Help Pop) */
.help-pop {
  position: relative;
  display: inline-flex;
  align-items: center;
}
.help-pop-btn {
  width: 18px;
  height: 18px;
  border-radius: 50%;
  border: 1px solid #d1d7e2;
  color: #6b7280;
  font-size: 12px;
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: default;
  background: #ffffff;
}
.help-pop-body {
  display: none;
  position: absolute;
  top: 22px;
  left: 0;
  min-width: 200px;
  background: #ffffff;
  border: 1px solid #e5e9f2;
  border-radius: 10px;
  padding: 8px 10px;
  font-size: 0.78rem;
  font-weight: 400;
  color: #4b5563;
  line-height: 1.4;
  box-shadow: 0 6px 16px rgba(17, 24, 39, 0.08);
  z-index: 100;
}
.help-pop:hover .help-pop-body {
  display: block;
}

/* Road List Styles */
.road-list { display: grid; gap: 10px; }
.road-item {
  background: #ffffff;
  border: 1px solid #e8ebf2;
  border-radius: 14px;
  padding: 10px 12px;
}
.road-item-title { font-weight: 800; margin-bottom: 4px; }
.road-item-meta { color: #666; font-size: 0.82rem; }
.road-tag {
  display: inline-block;
  margin-right: 6px;
  padding: 2px 8px;
  border-radius: 8px;
  background: #eef2ff;
  color: #2f5dff;
  font-size: 0.72rem;
  font-weight: 800;
}

div[data-testid="stPopover"] > button {
  background: #2f5dff;
  color: #fff;
  font-size: 0.72rem;
  font-weight: 800;
  padding: 4px 10px;
  border-radius: 999px;
  border: none;
}
</style>
"""

# -----------------------------
# 지도 높이 조정 CSS
# -----------------------------
MAP_HEIGHT_CSS = """
<style>
  .stFolium, .stFolium iframe {
    width: 100% !important;
    height: 360px !important;
    min-height: 360px !important;
  }
</style>
"""


def get_map_height_css(height: int = 360) -> str:
    """동적 지도 높이 CSS 생성."""
    return f"""
<style>
  .stFolium, .stFolium iframe {{
    width: 100% !important;
    height: {height}px !important;
    min-height: {height}px !important;
  }}
</style>
"""
