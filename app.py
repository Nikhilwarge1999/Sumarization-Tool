# import validators
# import streamlit as st
# from langchain.prompts import PromptTemplate
# from langchain_groq import ChatGroq
# from langchain.chains.summarize import load_summarize_chain
# from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader

# st.set_page_config(page_title="Youtube Summarizer", page_icon=":guardsman:", layout="wide")
# st.title("Langchain Summarizer")
# st.subheader("Summarize Youtube Videos and Web Pages")

# with st.sidebar:
#     st.subheader("GROQ API Key")
#     groq_api_key = st.text_input("Enter your GROQ API Key", type="password")

# generic_url = st.text_input("Enter the URL of the Youtube Video or Web Page", label_visibility="collapsed")

# llm = ChatGroq(groq_api_key=groq_api_key, model="llama3-70b-8192", temperature=0.1)

# prompt_template = """Provide the summary of the following content in 300 words or less.
# Content: {text}"""
# prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

# if st.button("Summerize the content from youtube or web"):
#     if not groq_api_key.strip() or not generic_url.strip():
#         st.error("Please enter a valid GROQ API Key")
#     elif not validators.url(generic_url):
#         st.error("Please enter a valid URL")
#     else:
#         try:
#             with st.spinner("Loading..."):
#                 if "youtube.com" in generic_url:
#                     loader = YoutubeLoader.from_youtube_url(generic_url, add_video_info=True)
#                 else:
#                     loader = UnstructuredURLLoader(
#                         urls=[generic_url],
#                         ssl_verify=False,
#                         headers={
#     "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
# }

#                     )
#                 documents = loader.load()
#                 st.success("Loaded the documents successfully")

#                 chain = load_summarize_chain(
#                     llm=llm,
#                     chain_type="stuff",  # try "map_reduce" if you face context issues
#                     prompt=prompt,
#                 )
#                 output_summary = chain.run(documents)
#                 st.success(output_summary)
#         except Exception as e:
#             st.exception(f"An error occurred: {e}")
import validators
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader

# --- Page Configuration ---
st.set_page_config(page_title="Youtube & Web Summarizer", page_icon="🎥", layout="wide")
st.title("🧠 Langchain Summarizer")
st.subheader("Summarize YouTube Videos and Web Pages with Groq + LangChain")

# --- Sidebar Inputs ---
with st.sidebar:
    st.header("🔐 API & Settings")
    groq_api_key = st.text_input("Enter your GROQ API Key", type="password")

    model = st.selectbox("Choose Groq Model", ["llama3-70b-8192", "mixtral-8x7b-32768"])
    temperature = st.slider("LLM Temperature", 0.0, 1.0, 0.1, 0.05)
    word_limit = st.slider("Summary Word Limit", 100, 500, 300, 50)
    show_preview = st.checkbox("Show original content preview", value=True)
    st.markdown("---")
    dark_mode = st.checkbox("🌙 Enable Dark Mode")

# --- Input for URL ---
generic_url = st.text_input("Enter a YouTube or Web Page URL")

# --- Track URL History in Session ---
if "url_history" not in st.session_state:
    st.session_state.url_history = []

# --- Initialize LLM ---
if groq_api_key:
    llm = ChatGroq(groq_api_key=groq_api_key, model=model, temperature=temperature)

prompt_template = f"""Provide the summary of the following content in {word_limit} words or less.
Content: {{text}}"""
prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

# --- Summarization Action ---
if st.button("📝 Summarize"):
    if not groq_api_key.strip() or not generic_url.strip():
        st.error("Please enter both a valid GROQ API Key and a URL.")
    elif not validators.url(generic_url):
        st.error("The URL format is invalid.")
    else:
        try:
            with st.spinner("🔄 Loading and summarizing content..."):
                # Loader selection
                if "youtube.com" in generic_url:
                    loader = YoutubeLoader.from_youtube_url(generic_url, add_video_info=True)
                else:
                    loader = UnstructuredURLLoader(
                        urls=[generic_url],
                        ssl_verify=False,
                        headers={"User-Agent": "Mozilla/5.0"}
                    )

                documents = loader.load()

                # Save URL to history
                if generic_url not in st.session_state.url_history:
                    st.session_state.url_history.insert(0, generic_url)
                    st.session_state.url_history = st.session_state.url_history[:5]  # Keep last 5

                if show_preview:
                    st.info("📄 Preview of the Original Content")
                    preview = documents[0].page_content[:800]
                    st.write(preview + "...")

                # Summarize
                chain = load_summarize_chain(llm=llm, chain_type="stuff", prompt=prompt)
                output_summary = chain.run(documents)

                st.success("✅ Summary Generated:")
                st.write(output_summary)

                # Download option
                st.download_button(
                    label="📥 Download Summary",
                    data=output_summary,
                    file_name="summary.txt",
                    mime="text/plain"
                )

        except Exception as e:
            st.exception(f"❌ An error occurred: {e}")

# --- URL History ---
if st.session_state.url_history:
    st.sidebar.markdown("---")
    st.sidebar.subheader("🕓 Recently Summarized URLs")
    for url in st.session_state.url_history:
        st.sidebar.markdown(f"- [{url}]({url})")

# --- Theme Setting (Streamlit handles CSS if needed externally) ---
if dark_mode:
    st.markdown(
        """
        <style>
        body {
            background-color: #0e1117;
            color: white;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
