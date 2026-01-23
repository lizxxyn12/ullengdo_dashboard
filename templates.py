"""
HTML 템플릿 생성 함수 모듈

이 모듈은 대시보드에서 사용되는 HTML 템플릿 생성 함수들을 포함합니다.
"""

from typing import Any


# -----------------------------
# 헤더 및 타이틀
# -----------------------------
def dashboard_title(logo_html: str = "") -> str:
    """대시보드 타이틀 HTML 생성."""
    return f"""
    <div class="dashboard-title">
        {logo_html}
        <div class="title-text">울릉도 데이터 대시보드</div>
    </div>
    """


def notice_pill(notice_html: str) -> str:
    """공지사항 pill HTML 생성."""
    return f'<div class="notice-pill">{notice_html}</div>'


def card_title(title: str) -> str:
    """카드 타이틀 HTML 생성."""
    return f'<div class="card-title">{title}</div>'


# -----------------------------
# 해상 공지 섹션
# -----------------------------
def sea_recent_events(
    arrive_passengers: int,
    arrive_dt_label: str,
    arrive_ship_name: str,
    arrive_vehicles: str,
    depart_passengers: int,
    depart_dt_label: str,
    depart_ship_name: str,
    depart_vehicles: str,
    arrive_avg3_passengers: int,
    arrive_avg3_vehicles: str,
    depart_avg3_passengers: int,
    depart_avg3_vehicles: str,
) -> str:
    """최근 이벤트 섹션 HTML 생성."""
    return f"""
<div class="sea-section">
  <div class="sea-section-title">가장 최근 이벤트</div>
  <div class="sea-kpi-grid">
    <div class="sea-kpi-card">
      <div class="sea-kpi-title">최근 입항 1건</div>
      <div class="sea-kpi-value">{arrive_passengers:,}명</div>
      <div class="sea-kpi-meta">
        일시: {arrive_dt_label}<br/>
        선박명: {arrive_ship_name}<br/>
        입항차량수: {arrive_vehicles}
      </div>
    </div>
    <div class="sea-kpi-card">
      <div class="sea-kpi-title">최근 출항 1건</div>
      <div class="sea-kpi-value">{depart_passengers:,}명</div>
      <div class="sea-kpi-meta">
        일시: {depart_dt_label}<br/>
        선박명: {depart_ship_name}<br/>
        출항차량수: {depart_vehicles}
      </div>
    </div>
    <div class="sea-kpi-card" style="grid-column: 1 / -1;">
      <div class="sea-kpi-title">최근 3회 평균</div>
      <div class="sea-kpi-meta">
        평균 입항객수: {arrive_avg3_passengers:,}명 · 평균 입항차량수: {arrive_avg3_vehicles}<br/>
        평균 출항객수: {depart_avg3_passengers:,}명 · 평균 출항차량수: {depart_avg3_vehicles}
      </div>
    </div>
  </div>
</div>
    """


def sea_monthly_stats(
    period_label: str,
    monthly_arrive_ship: int,
    monthly_depart_ship: int,
    arrive_sum: int,
    arrive_vehicle_sum: str,
    depart_sum: int,
    depart_vehicle_sum: str,
    badge_html: str,
) -> str:
    """월간 통계 섹션 HTML 생성."""
    return f"""
<div class="sea-section">
  <div class="sea-section-title">최근 30일 기준</div>
  <div class="sea-kpi-grid">
    <div class="sea-kpi-card">
      <div class="sea-kpi-title">기간</div>
      <div class="sea-kpi-meta">{period_label}</div>
    </div>
    <div class="sea-kpi-card">
      <div class="sea-kpi-title">월간 선박 수</div>
      <div class="sea-kpi-meta">입항 {monthly_arrive_ship}건 · 출항 {monthly_depart_ship}건</div>
    </div>
    <div class="sea-kpi-card">
      <div class="sea-kpi-title">월간 입항객수 합계</div>
      <div class="sea-kpi-value">{arrive_sum:,}명</div>
      <div class="sea-kpi-meta">입항차량수 합계: {arrive_vehicle_sum}</div>
    </div>
    <div class="sea-kpi-card">
      <div class="sea-kpi-title">월간 출항객수 합계</div>
      <div class="sea-kpi-value">{depart_sum:,}명</div>
      <div class="sea-kpi-meta">출항차량수 합계: {depart_vehicle_sum}</div>
    </div>
  </div>
  <div style="margin-top: 10px;">
    {badge_html}
  </div>
</div>
    """


def sea_badges(badges: list[str]) -> str:
    """배지 HTML 생성."""
    if badges:
        badge_items = "".join([f"<span class='sea-badge'>{b}</span>" for b in badges])
        return f"<div class='sea-badges'>{badge_items}</div>"
    return "<div class='sea-badges'><span class='sea-badge'>이번 달 이슈 없음</span></div>"


def bar_row(
    label: str,
    sub_label: str,
    help_text: str,
    value: int,
    pct: float,
    color: str,
) -> str:
    """막대 그래프 행 HTML 생성."""
    return f"""
<div class="bar-row">
  <div class="bar-label">
    <div class="bar-label-wrap">
      <span>{label}</span>
      <span class="bar-sub">{sub_label}</span>
      <span class="help-pop">
        <span class="help-pop-btn">?</span>
        <span class="help-pop-body">
          {help_text}
        </span>
      </span>
    </div>
  </div>
  <div class="bar-track">
    <div class="bar-fill" style="width:{pct}%; background:{color};">
      <div class="bar-value-onfill">{value:,}</div>
    </div>
  </div>
</div>
    """


def sea_yearly_stats(
    sea_arrive: int,
    sea_arrive_pct: float,
    sea_arrive_people: int,
    sea_depart: int,
    sea_depart_pct: float,
    sea_depart_people: int,
    sea_arrive_ship_total: int,
    sea_arrive_ship_pct: float,
    sea_depart_ship_total: int,
    sea_depart_ship_pct: float,
    sea_control: int,
    sea_control_pct: float,
    sea_cancel: int,
    sea_cancel_pct: float,
    sea_change: int,
    sea_change_pct: float,
) -> str:
    """연간 통계 섹션 HTML 생성."""
    bars_html = ""

    # 입항
    bars_html += bar_row(
        "입항", "(배 당 입도객 평균)",
        f"배 당 여객 입항 평균: <b>{sea_arrive_people:,}명</b>",
        sea_arrive, sea_arrive_pct, "#ffd3a8"
    )

    # 출항
    bars_html += bar_row(
        "출항", "(배 당 출도객 평균)",
        f"배 당 여객 출항 평균: <b>{sea_depart_people:,}명</b>",
        sea_depart, sea_depart_pct, "#8fe3da"
    )

    # 입항 선박 수
    bars_html += bar_row(
        "입항 선박 수", "(합계)",
        f"2025년 입항 선박 합계: <b>{sea_arrive_ship_total:,}건</b>",
        sea_arrive_ship_total, sea_arrive_ship_pct, "#ff8a3d"
    )

    # 출항 선박 수
    bars_html += bar_row(
        "출항 선박 수", "(합계)",
        f"2025년 출항 선박 합계: <b>{sea_depart_ship_total:,}건</b>",
        sea_depart_ship_total, sea_depart_ship_pct, "#00b3a4"
    )

    # 운항통제
    bars_html += bar_row(
        "운항통제", "(합계)",
        f"2025년 기상 악화 등으로 통제된 선박 수입니다.<br/>배 운항통제 합계: {sea_control:,}건",
        sea_control, sea_control_pct, "#5b2bff"
    )

    # 결항
    bars_html += bar_row(
        "결항", "(합계)",
        f"2025년 기상 또는 점검 사유로 취소된 선박 수입니다.<br/>배 결항 합계: {sea_cancel:,}건",
        sea_cancel, sea_cancel_pct, "#e24a4a"
    )

    # 시간변경
    bars_html += bar_row(
        "시간변경", "(합계)",
        f"2025년 출항/입항 시간이 변경된 선박 수입니다.<br/>배 시간변경 합계: {sea_change:,}건",
        sea_change, sea_change_pct, "#7b61ff"
    )

    return f"""
<div class="r2-card">
<div class="sea-section">
<div class="sea-section-title">연간 통계 (2025년 기준)</div>
<div class="sea-bars">
{bars_html}
</div>
</div>
</div>
"""


# -----------------------------
# 도로 통제 섹션
# -----------------------------
def road_control_header() -> str:
    """도로 통제 헤더 HTML 생성."""
    return """
<div class="r2-top">
  <div class="r2-title">도로 통제 공지</div>
  <div class="r2-date">최신 기준</div>
</div>
    """


def road_control_card_start() -> str:
    """도로 통제 카드 시작 HTML."""
    return """
<div class="r2-card r2-card-body">
  <div class="road-list">
    """


def road_control_card_end() -> str:
    """도로 통제 카드 종료 HTML."""
    return """
  </div>
</div>
    """


def road_item(tag: str, title: str, meta: str) -> str:
    """도로 통제 아이템 HTML 생성."""
    return f"""
    <div class="road-item">
      <span class="road-tag">{tag}</span>
      <span class="road-item-title">{title}</span>
      <div class="road-item-meta">{meta}</div>
    </div>
    """


# -----------------------------
# 버스 정류장 섹션
# -----------------------------
def bus_route_grid(routes_html: str) -> str:
    """버스 노선 그리드 HTML 생성."""
    return f'<div class="bus-route-grid">{routes_html}</div>'


def bus_route_card(route_id: str, route_desc: str, color: str) -> str:
    """버스 노선 카드 HTML 생성."""
    return f"""
<div class="bus-route-card" style="border-left-color: {color};">
  <div class="bus-route-id">{route_id}</div>
  <div class="bus-route-desc">{route_desc}</div>
</div>
    """


def bus_route_empty() -> str:
    """버스 노선 없음 HTML 생성."""
    return '<div class="bus-route-empty">경유 노선 정보가 없습니다.</div>'


# -----------------------------
# 사진 플레이스홀더
# -----------------------------
def photo_placeholder(message: str = "사진 없음", sub_message: str = "") -> str:
    """사진 플레이스홀더 HTML 생성."""
    sub_html = f"<br/><span style='font-size:0.8rem;'>{sub_message}</span>" if sub_message else ""
    return f"""
<div class="photo-placeholder">
  {message}{sub_html}
</div>
    """


def photo_loading_placeholder() -> str:
    """사진 로딩 플레이스홀더 HTML 생성."""
    return """
<div style="background:#f0f2f6; height:86px; border-radius:10px; display:flex; align-items:center; justify-content:center; color:#888; font-size:0.85rem;">
  사진 로딩 중...
</div>
    """


# -----------------------------
# 선택 표시
# -----------------------------
def selected_tag() -> str:
    """선택 표시 태그 HTML 생성."""
    return " <span style='color:#d12c2c;'>● 선택</span>"


def meta_info(text: str) -> str:
    """메타 정보 HTML 생성."""
    return f"<div style='color:#666; font-size:0.85rem;'>{text}</div>"
