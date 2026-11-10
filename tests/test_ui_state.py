"""화면에서 선택 위젯이 제거됐다 다시 나타나는 실제 Streamlit 흐름을 검증한다."""

from streamlit.testing.v1 import AppTest


APP = '''
import streamlit as st
from modules.ui_state import remembered_radio, remembered_selectbox

page = st.radio("분석 종류", ["비교 분석", "진동 확인", "CSV 진단"], key="page")
if page == "비교 분석":
    source = remembered_radio("비교할 기록", ["설정 조정용 기록", "새 MAT 업로드"],
                              "vibration_comparison_source")
    if source == "설정 조정용 기록":
        remove = st.checkbox("두 번째 파일 제거")
        files = ["K003.mat"] if remove else ["K003.mat", "KI04.mat"]
        remembered_selectbox("MAT 기록", files, "vibration_comparison_file")
    else:
        st.info("업로드 화면")
elif page == "진동 확인":
    remembered_selectbox("확인할 MAT 파일", ["K001.mat", "KA04.mat"], "vibration_inspection_file")
st.button("다시 실행")
'''


def test_each_mat_selection_survives_menu_switches_and_reruns():
    app = AppTest.from_string(APP).run()
    app.selectbox[0].set_value('KI04.mat').run()
    app.radio(key='page').set_value('진동 확인').run()
    app.selectbox[0].set_value('KA04.mat').run()
    app.radio(key='page').set_value('CSV 진단').run()
    # 두 화면 모두 실제로 렌더링하지 않은 실행을 거친 뒤 각각의 파일을 복원해야 한다.
    app.radio(key='page').set_value('비교 분석').run()
    assert app.selectbox[0].value == 'KI04.mat'
    app.button[0].click().run()
    assert app.selectbox[0].value == 'KI04.mat'
    app.radio(key='page').set_value('진동 확인').run()
    assert app.selectbox[0].value == 'KA04.mat'
    assert not app.exception


def test_source_switch_remembers_source_and_previous_mat_file():
    app = AppTest.from_string(APP).run()
    app.selectbox[0].set_value('KI04.mat').run()
    app.radio(key='_vibration_comparison_source').set_value('새 MAT 업로드').run()
    assert len(app.selectbox) == 0
    app.radio(key='page').set_value('진동 확인').run()
    app.radio(key='page').set_value('비교 분석').run()
    assert app.radio(key='_vibration_comparison_source').value == '새 MAT 업로드'
    app.radio(key='_vibration_comparison_source').set_value('설정 조정용 기록').run()
    assert app.selectbox[0].value == 'KI04.mat'
    assert not app.exception


def test_removed_file_falls_back_to_an_available_record():
    app = AppTest.from_string(APP).run()
    app.selectbox[0].set_value('KI04.mat').run()
    app.checkbox[0].check().run()
    assert app.selectbox[0].value == 'K003.mat'
    app.radio(key='page').set_value('CSV 진단').run()
    app.radio(key='page').set_value('비교 분석').run()
    assert app.selectbox[0].value == 'K003.mat'
    assert not app.exception
