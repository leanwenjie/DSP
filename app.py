"""Streamlit demonstration of frozen CLIP-based misinformation screening."""
import sys
from pathlib import Path
import streamlit as st
sys.path.insert(0, str(Path(__file__).resolve().parent/'scripts'))
from app_inference import MODELS, Detector, validate_input

st.set_page_config(page_title='Image & Caption Review', page_icon='🔎', layout='centered')
st.title('Image & Caption Review')
st.write('Screen an image and its caption for a possible misleading pairing.')
st.info('This model cannot verify the event, date, location or factual truth. A result is a signal for human review.')

@st.cache_resource(show_spinner=False)
def load_detector():
    return Detector()

with st.form('assessment'):
    upload = st.file_uploader('Upload an image', type=['jpg','jpeg','png','webp'], help='Still images up to 10 MB and 20 megapixels.')
    caption = st.text_area('Caption', placeholder='Enter the caption you want to assess.', max_chars=4000)
    model_name = st.selectbox('Model', MODELS)
    submitted = st.form_submit_button('Assess pairing', type='primary')

if submitted:
    try:
        content = upload.getvalue() if upload is not None else b''
        image, text = validate_input(content, caption)
        st.image(image, caption='Submitted image', width='stretch')
        st.write('**Submitted caption:**')
        st.text(text)
        with st.spinner('Assessing the pairing. The first run loads the model and may take longer…'):
            result = load_detector().predict(image, text, model_name)
        if result['flagged']:
            st.warning('Flagged for review — the model found signs of a misleading pairing.')
        else:
            st.info('No mismatch flagged — this does not confirm that the caption is true.')
        left, right = st.columns(2)
        left.metric('Model score', f"{result['score']:.4f}")
        right.metric('Analysis time', f"{result['seconds']:.2f} s")
        st.caption('Analysis time excludes initial model loading, upload and image validation.')
        st.write('Check the original source, publication date and location before relying on this content.')
        with st.expander('Understand this result'):
            st.write(f"Model: {result['model']}. Score type: {result['score_type']}.")
            st.write(f"This model flags a pair when the score is {result['flag_when']} its fixed threshold ({result['threshold']:.6f}). The comparison uses full precision.")
            st.write('The score is not a probability that the claim is false. The model uses learned image–caption relationships; it does not search for external evidence or identify a specific factual error.')
    except ValueError as exc:
        st.error(str(exc))
    except (RuntimeError, OSError):
        st.error('Inference is unavailable. Check the model files and runtime using the project setup instructions, then try again.')

with st.expander('Research context and limitations'):
    st.write('Bachelor’s project: Multimodal Out-of-Context Misinformation Detection using Vision-Language Models.')
    st.write('The default model uses frozen CLIP ViT-L/14 features and logistic regression. The other options retain the earlier B/32 experiments. Classifiers were trained on NewsCLIPpings; thresholds were selected on validation data and kept unchanged for NewsCLIPpings test and VERITE evaluation.')
    st.write('The default L/14 model reached 83.62% macro-F1 on NewsCLIPpings test. On VERITE it reached 73.88% macro-F1 for true versus out-of-context pairs, detecting 55.69% of OOC pairs; macro-F1 was 53.77% when miscaptioned pairs were also included. These benchmark results do not guarantee performance on your upload.')
    st.write('Uploads and captions are processed by this app’s server in memory. The app does not write them to disk or send them to an external prediction API. Use a trusted local server for private content.')
