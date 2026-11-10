"""조건부로 숨겨지는 선택 상자의 값을 같은 브라우저 세션에서 기억한다.

Streamlit은 화면에 없는 위젯의 상태를 정리한다. 위젯용 키와 보관용 키를
분리하면 다른 메뉴로 갔다 돌아와도 마지막 선택을 복원할 수 있다.
"""

from collections.abc import Callable, Sequence

import streamlit as st


def _save_choice(state_key: str, widget_key: str) -> None:
    """선택이 바뀌는 즉시 위젯과 연결되지 않은 보관용 상태에 복사한다."""
    st.session_state[state_key] = st.session_state[widget_key]


def _remembered_choice(widget: Callable, label: str, options: Sequence[str], state_key: str) -> str:
    if not options:
        raise ValueError("선택 가능한 항목이 없습니다.")
    saved = st.session_state.get(state_key)
    if saved not in options:
        saved = options[0]  # 이전 파일이 목록에서 제거됐다면 유효한 첫 항목을 사용한다.
    widget_key = f"_{state_key}"
    st.session_state[widget_key] = saved
    choice = widget(label, options, key=widget_key, on_change=_save_choice,
                    args=(state_key, widget_key))
    st.session_state[state_key] = choice
    return choice


def remembered_selectbox(label: str, options: Sequence[str], state_key: str) -> str:
    """파일명처럼 변하지 않는 문자열을 저장하며, 임시 Path나 목록 순서에 의존하지 않는다."""
    return _remembered_choice(st.sidebar.selectbox, label, options, state_key)


def remembered_radio(label: str, options: Sequence[str], state_key: str) -> str:
    return _remembered_choice(st.sidebar.radio, label, options, state_key)
