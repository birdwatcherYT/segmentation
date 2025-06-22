import streamlit as st
from PIL import Image
import numpy as np
import io

# streamlit_image_coordinates は画像をインタラクティブにするための外部ライブラリです。
from streamlit_image_coordinates import streamlit_image_coordinates

# Ultralytics SAMモデルをインポート
from ultralytics import SAM


# --- SAMモデルの初期化 ---
# この関数はSAMモデルをロードし、Streamlitのキャッシュ機能を使って
# アプリケーションの再実行時にもモデルが再度ロードされないようにします。
@st.cache_resource
def load_sam_model():
    """
    Ultralytics SAM 2モデルをロードします。
    モデルファイルはUltralyticsライブラリによって自動的にダウンロードされます。
    """
    try:
        # SAM 2モデルのロード。ここでは 'sam2.1_b.pt' を使用します。
        # 他のモデル（例: 'sam_b.pt' for original SAM, 'sam2_s.pt' for smaller SAM 2）も利用可能です。
        # Ultralyticsはモデルを自動的にダウンロードします。
        model = SAM("sam2.1_b.pt")
        st.success("Ultralytics SAM 2モデルが正常にロードされました。")
        return model
    except Exception as e:
        st.error(f"Ultralytics SAM 2モデルのロード中にエラーが発生しました: {e}")
        st.error("`pip install ultralytics` が正常に実行されているか確認してください。")
        st.stop()


# モデルをロード
# `predictor`という変数名をそのまま使うが、実体はUltralyticsのSAMモデルインスタンス
predictor = load_sam_model()

# --- Streamlit UIの構築 ---
st.set_page_config(layout="wide", page_title="画像セグメンテーション（SAM 2使用）")

st.title("画像セグメンテーションアプリケーション")

st.markdown(
    """
    このアプリケーションは、**Ultralytics版のSegment Anything Model 2 (SAM 2)** を使用して、
    画像から指定したオブジェクトを切り出し、透過PNGとして出力します。
    
    ---
    **使い方:**
    1.  上の「画像をアップロードしてください」から画像をアップロードします。
    2.  画像が表示されたら、切り出したいオブジェクトの中心付近を**クリック**します。
        * **複数点を選択**: 異なる場所を複数回クリックすると、複数の点が蓄積されます。
        * **点の選択をリセット**: 「点の選択をリセット」ボタンをクリックすると、選択した点がすべてクリアされます。
    3.  点を選択した後、「セグメンテーションを実行」ボタンをクリックしてセグメンテーションを開始します。
    4.  セグメンテーションが実行され、結果が下に表示されます。
    5.  「透過PNGをダウンロード」ボタンから結果の画像をダウンロードできます。
"""
)

# st.session_stateに点のリストとセグメンテーション結果、処理中フラグを初期化
if "points" not in st.session_state:
    st.session_state.points = []
if "image_uploaded_name" not in st.session_state:
    st.session_state.image_uploaded_name = None
if (
    "segmented_image_rgba" not in st.session_state
):  # セグメンテーション結果を保持する新しい状態変数
    st.session_state.segmented_image_rgba = None
if "is_processing" not in st.session_state:  # 処理中かどうかを示すフラグ
    st.session_state.is_processing = False
if (
    "reset_counter" not in st.session_state
):  # streamlit_image_coordinates のキーをリセットするためのカウンター
    st.session_state.reset_counter = 0

# ファイルアップローダー
uploaded_file = st.file_uploader(
    "画像をアップロードしてください", type=["png", "jpg", "jpeg"]
)

# 新しい画像がアップロードされた場合、点とセグメンテーション結果、処理中フラグをリセット
if (
    uploaded_file is not None
    and st.session_state.image_uploaded_name != uploaded_file.name
):
    st.session_state.points = []
    st.session_state.segmented_image_rgba = None  # 新しい画像で結果をリセット
    st.session_state.image_uploaded_name = uploaded_file.name
    st.session_state.is_processing = False  # 新しい画像で処理中フラグをリセット
    st.session_state.reset_counter = 0  # 新しい画像でカウンターもリセット

if uploaded_file is not None:
    # アップロードされた画像をPIL Imageとして読み込み、RGB形式に変換
    image = Image.open(uploaded_file).convert("RGB")
    # Ultralyticsモデルが処理できるようにPIL ImageをNumPy配列に変換
    image_np = np.array(image)

    st.subheader(
        "画像をプレビューして、切り出したいオブジェクトの中心をクリックしてください"
    )

    # streamlit_image_coordinatesを使用して、画像上のクリック座標を取得します。
    # `key`をユニークにして、新しい画像がアップロードされたときにクリック状態がリセットされるようにします。
    # reset_counter をキーに含めることで、リセット時に widget が再描画されるようにする
    value = streamlit_image_coordinates(
        image, key=f"image_coords_{uploaded_file.name}_{st.session_state.reset_counter}"
    )

    # 新しいクリックがあれば点を追加
    if value:
        new_x, new_y = value["x"], value["y"]
        # クリックされた点が既にリストにない場合のみ追加（同じ点を複数回追加しないように）
        if [new_x, new_y] not in st.session_state.points:  # 直接リストの要素として比較
            st.session_state.points.append([new_x, new_y])
            st.session_state.segmented_image_rgba = (
                None  # 新しい点が追加されたら既存のセグメンテーション結果をクリア
            )
            st.session_state.is_processing = (
                False  # 新しい点が追加されたら処理中フラグをリセット
            )
            st.rerun()  # 点が追加されたらUIを更新して、現在の点のリストを表示

    # 現在選択されている点を表示
    if st.session_state.points:
        st.info(f"選択された点: {st.session_state.points}")
        col1, col2 = st.columns(2)
        with col1:
            # 点をクリアするボタン
            if st.button("点の選択をリセット"):
                st.session_state.points = []
                st.session_state.segmented_image_rgba = (
                    None  # 点をリセットしたらセグメンテーション結果もリセット
                )
                st.session_state.is_processing = (
                    False  # 点をリセットしたら処理中フラグをリセット
                )
                st.session_state.reset_counter += (
                    1  # カウンターをインクリメントして widget のキーを変更
                )
                st.rerun()  # 点がリセットされたらUIを更新
        with col2:
            # セグメンテーション実行ボタン
            if st.button("セグメンテーションを実行"):
                # セグメンテーション処理を開始するフラグを立てる
                st.session_state.is_processing = True
                st.session_state.segmented_image_rgba = (
                    None  # 実行前に既存の結果をクリア
                )
                st.rerun()  # スピナーを表示するために即座に再実行

    else:
        st.info(
            "画像をプレビューして、切り出したいオブジェクトの中心をクリックしてください。"
        )
else:
    st.info("画像をアップロードして開始してください。")

# --- セグメンテーション結果の表示 ---
# 処理中フラグが立っている場合、スピナーを表示し、セグメンテーションを実行
if st.session_state.is_processing and uploaded_file is not None:
    # 既存の画像データが必要なので、ここでもう一度読み込む（rerunで状態がリセットされるため）
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    with st.spinner("セグメンテーションを実行中...しばらくお待ちください。"):
        try:
            input_points = np.array(st.session_state.points)
            input_labels = np.ones(len(st.session_state.points))  # すべて前景ラベル (1)

            # Ultralytics SAM 2モデルを使用してマスクを予測します。
            results = predictor(
                source=image_np, points=input_points, labels=input_labels
            )

            if results and results[0].masks and results[0].masks.data is not None:
                mask = results[0].masks.data[0].cpu().numpy()

                segmented_image_rgba = Image.new("RGBA", image.size)
                segmented_image_rgba.paste(image, mask=Image.fromarray(mask))

                st.session_state.segmented_image_rgba = segmented_image_rgba
            else:
                st.warning(
                    "指定された点ではセグメンテーションマスクが見つかりませんでした。別の点を試してください。"
                )
                st.session_state.segmented_image_rgba = None  # 結果がない場合はクリア
        except Exception as e:
            st.error(f"セグメンテーション中にエラーが発生しました: {e}")
            st.error("モデルの入力点や画像の形式を確認してください。")
            st.warning(
                "大規模な画像をセグメンテーションする場合、メモリが不足する可能性があります。"
            )
        finally:
            st.session_state.is_processing = False  # 処理完了後にフラグをリセット
            st.rerun()  # スピナーを消して結果を表示するために再実行

# セグメンテーション結果がセッションステートに保存されている場合のみ表示
if st.session_state.segmented_image_rgba is not None:
    st.subheader("セグメンテーション結果（透過PNG）")
    st.image(
        st.session_state.segmented_image_rgba,
        caption="セグメントされた画像",
        use_container_width=True,
    )

    # 透過PNGとしてダウンロードできるようにBytesIOに保存
    img_byte_arr = io.BytesIO()
    st.session_state.segmented_image_rgba.save(img_byte_arr, format="PNG")
    img_byte_arr.seek(0)

    # ダウンロードボタン
    st.download_button(
        label="透過PNGをダウンロード",
        data=img_byte_arr,
        file_name=f"segmented_{uploaded_file.name.split('.')[0]}.png",
        mime="image/png",
    )
