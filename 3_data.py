import streamlit as st
from PIL import Image, ImageDraw
import numpy as np
from sklearn.cluster import KMeans
from streamlit_drawable_canvas import st_canvas
import os
import zipfile
import io

# 페이지 설정
st.set_page_config(layout="wide")

st.title("이미지 클러스터링 및 st라벨링 도구")

# 1. 이미지 업로드
uploaded_files = st.file_uploader(
    "이미지를 업로드하세요 (여러 개 선택 가능)",
    type=["bmp", "jpg", "png", "jpeg"],
    accept_multiple_files=True
)

if uploaded_files:
    # 이미지 이름 리스트 생성
    image_names = [uploaded_file.name for uploaded_file in uploaded_files]

    # 이전에 선택한 이미지 이름을 세션 상태에 저장 (초기화)
    if 'prev_selected_image_name' not in st.session_state:
        st.session_state.prev_selected_image_name = None

    # 사이드바에 이미지 목록 표시
    st.sidebar.title("이미지 목록")
    selected_image_name = st.sidebar.selectbox(
        "이미지를 선택하세요",
        options=image_names,
        index=0
    )

    # 선택된 이미지가 변경되었는지 확인하고, 변경되었으면 세션 상태 초기화
    if st.session_state.prev_selected_image_name != selected_image_name:
        st.session_state.prev_selected_image_name = selected_image_name
        # 세션 상태 초기화
        st.session_state.selected_points = []
        st.session_state.labels = None
        st.session_state.clustered_image = None
        st.session_state.unique_labels = None
        st.session_state.background_labels = []
        # 이미지별로 mask_images를 관리하도록 수정
        if 'mask_images' not in st.session_state:
            st.session_state.mask_images = {}
        # 현재 이미지의 mask_image를 초기화
        if selected_image_name in st.session_state.mask_images:
            del st.session_state.mask_images[selected_image_name]

    # 선택된 이미지 가져오기
    selected_image_file = next((file for file in uploaded_files if file.name == selected_image_name), None)

    if selected_image_file:
        # 이미지 로드
        image = Image.open(selected_image_file).convert("RGB")
        image_np = np.array(image)

        # 이미지 크기 조절 (필요에 따라)
        max_width = 800
        if image.width > max_width:
            ratio = max_width / image.width
            new_size = (max_width, int(image.height * ratio))
            image = image.resize(new_size)
            image_np = np.array(image)

        # 선택된 이미지 표시
        st.subheader("선택된 이미지")
        st.image(image, use_column_width=True)

        # 좌표 저장을 위한 세션 상태 초기화 (이미 초기화됨)
        if 'selected_points' not in st.session_state:
            st.session_state.selected_points = []
        if 'labels' not in st.session_state:
            st.session_state.labels = None
        if 'clustered_image' not in st.session_state:
            st.session_state.clustered_image = None
        if 'unique_labels' not in st.session_state:
            st.session_state.unique_labels = None
        if 'background_labels' not in st.session_state:
            st.session_state.background_labels = []
        if 'mask_images' not in st.session_state:
            st.session_state.mask_images = {}

        # 2. 좌표 선택을 위한 캔버스 생성
        st.subheader("좌표 선택")

        # 캔버스 생성 (이미지 크기에 맞게)
        canvas_result = st_canvas(
            fill_color="rgba(255, 0, 0, 0.3)",  # 반투명 빨간색
            stroke_width=10,
            background_image=image,
            update_streamlit=True,
            height=image.height,
            width=image.width,
            drawing_mode="point",
            point_display_radius=5,
            key=f"select_point_{selected_image_name}",  # 이미지별로 키를 다르게 설정
        )

        # 캔버스 결과 처리
        if canvas_result.json_data is not None:
            objects = canvas_result.json_data["objects"]
            if objects:
                # 모든 클릭한 좌표 저장
                st.session_state.selected_points = []
                for obj in objects:
                    if obj["type"] == "circle":
                        x = int(obj["left"] + obj["radius"])
                        y = int(obj["top"] + obj["radius"])
                        st.session_state.selected_points.append((x, y))

                # 선택한 좌표를 이미지에 표시
                image_with_points = image.copy()
                draw = ImageDraw.Draw(image_with_points)
                for idx, point in enumerate(st.session_state.selected_points):
                    radius = 5
                    x, y = point
                    draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill='red')
                    draw.text((x + radius, y - radius), str(idx + 1), fill='red')
                st.image(image_with_points, caption='클릭한 좌표가 표시된 이미지', use_column_width=True)

                st.success(f"선택된 좌표들: {st.session_state.selected_points}")
            else:
                st.session_state.selected_points = []
                st.warning("이미지 위에서 좌표를 클릭하세요.")
        else:
            st.warning("이미지 위에서 좌표를 클릭하세요.")

        # 리셋 버튼
        if st.button("리셋"):
            st.session_state.selected_points = []
            st.session_state.labels = None
            st.session_state.clustered_image = None
            st.session_state.unique_labels = None
            st.session_state.background_labels = []
            # 현재 이미지의 mask_image를 초기화
            if selected_image_name in st.session_state.mask_images:
                del st.session_state.mask_images[selected_image_name]
            st.experimental_rerun()  # 앱 재실행

        # 제출 버튼
        if st.button("제출"):
            if st.session_state.selected_points:
                # 선택한 좌표의 RGB 값을 초기 중심값으로 KMeans 클러스터링 수행
                pixels = image_np.reshape(-1, 3)
                init_colors = np.array([image_np[y, x] for x, y in st.session_state.selected_points])

                n_clusters = len(st.session_state.selected_points)
                kmeans = KMeans(n_clusters=n_clusters, init=init_colors, n_init=1)
                kmeans.fit(pixels)

                # 라벨 얻기
                labels = kmeans.labels_
                label_array = labels.reshape(image_np.shape[:2])
                st.session_state.labels = label_array
                st.session_state.clustered_image = kmeans.cluster_centers_[labels].reshape(image_np.shape).astype(np.uint8)
                st.session_state.unique_labels = np.unique(label_array)
                st.success("클러스터링이 완료되었습니다.")
            else:
                st.warning("좌표를 선택한 후 제출 버튼을 눌러주세요.")

        # 클러스터링 결과가 있으면
        if 'labels' in st.session_state and st.session_state.labels is not None:
            # 클러스터링된 이미지 표시
            st.subheader("클러스터링된 이미지")
            st.image(st.session_state.clustered_image, use_column_width=True)

            # 사이드바에 라벨 목록 표시 및 체크박스로 배경 라벨 선택
            st.sidebar.subheader("라벨 목록 및 배경 라벨 선택")
            background_labels = []
            for label in st.session_state.unique_labels:
                checked = st.sidebar.checkbox(f"라벨 {label}", key=f"label_{label}_{selected_image_name}")
                if checked:
                    background_labels.append(label)
            st.session_state.background_labels = background_labels

            # 선택한 라벨을 오버레이로 표시하는 기능 추가
            st.subheader("라벨 오버레이")
            selected_label_for_overlay = st.sidebar.selectbox("오버레이로 표시할 라벨을 선택하세요", st.session_state.unique_labels, key=f"overlay_label_{selected_image_name}")
            mask = st.session_state.labels == selected_label_for_overlay
            overlay = Image.new('RGBA', image.size, (255, 0, 0, 100))  # 붉은색 오버레이
            base_image = image.convert('RGBA')
            combined_image = Image.composite(overlay, base_image, Image.fromarray(mask.astype('uint8') * 255))
            st.image(combined_image, caption=f'라벨 {selected_label_for_overlay} 오버레이', use_column_width=True)

            # 라벨 수정 안내
            st.subheader("라벨 수정")
            st.write("마우스로 클릭하거나 드래그하여 라벨을 변경할 영역을 선택하세요.")

            # 라벨 수정 캔버스 생성
            canvas_result_edit = st_canvas(
                fill_color="rgba(0, 255, 0, 0.5)",  # 반투명 녹색
                stroke_width=10,
                background_image=combined_image,
                update_streamlit=True,
                height=image.height,
                width=image.width,
                drawing_mode="freedraw",
                key=f"edit_label_{selected_image_name}",
            )

            # 수정할 라벨 선택
            new_label = st.number_input("새로운 라벨 번호를 입력하세요", min_value=0, max_value=100, value=int(selected_label_for_overlay), key=f"new_label_{selected_image_name}")

            # 완료 버튼 클릭 시 라벨 업데이트
            if st.button("완료"):
                if canvas_result_edit.image_data is not None:
                    if canvas_result_edit.image_data.shape[:2] != st.session_state.labels.shape:
                        st.error("캔버스의 크기와 라벨 배열의 크기가 일치하지 않습니다.")
                    else:
                        edited_mask = canvas_result_edit.image_data[:, :, 3] > 0  # 알파 채널이 0보다 큰 영역
                        st.session_state.labels[edited_mask] = new_label  # 선택된 라벨로 변경
                        st.success("선택된 영역의 라벨이 변경되었습니다.")
                else:
                    st.warning("수정할 영역을 선택하세요.")

            # 배경 라벨은 0으로, 나머지는 1로 변경
            final_labels = np.where(np.isin(st.session_state.labels, st.session_state.background_labels), 0, 1)

            # 라벨 결과를 mask.png로 저장
            mask_image = Image.fromarray((final_labels * 255).astype('uint8'))
            st.session_state.mask_images[selected_image_name] = mask_image

            st.subheader("최종 라벨 결과")
            st.image(mask_image, caption='mask.png', use_column_width=True)

            # 모든 이미지에 대한 라벨이 생성되었으면 labels.zip으로 제공
            if st.button("라벨 저장"):
                if st.session_state.mask_images:
                    # 임시 버퍼 생성
                    zip_buffer = io.BytesIO()
                    with zipfile.ZipFile(zip_buffer, "w") as zip_file:
                        for name, mask_img in st.session_state.mask_images.items():
                            # mask.png로 저장
                            img_byte_arr = io.BytesIO()
                            mask_img.save(img_byte_arr, format='PNG')
                            img_byte_arr = img_byte_arr.getvalue()
                            # 파일명을 이미지 이름에 맞게 설정
                            zip_file.writestr(f"{os.path.splitext(name)[0]}_mask.png", img_byte_arr)
                    zip_buffer.seek(0)
                    st.download_button(
                        label="labels.zip 다운로드",
                        data=zip_buffer,
                        file_name="labels.zip",
                        mime="application/zip"
                    )
                else:
                    st.warning("라벨이 생성되지 않았습니다. 먼저 라벨을 생성하세요.")
