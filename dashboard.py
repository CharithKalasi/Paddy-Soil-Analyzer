import os

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

from predict import phase1_predict, phase2_predict


st.set_page_config(page_title="KrishiLink Soil AI Dashboard", page_icon="🌾", layout="wide")
load_dotenv()


def flatten_recommendations(result: dict) -> dict:
    flat_result = {}
    for section, values in result.items():
        if isinstance(values, dict):
            for key, value in values.items():
                flat_result[key] = round(float(value), 2)
        else:
            flat_result[section] = round(float(values), 2)
    return flat_result


def render_vertical_list(title: str, values: dict) -> None:
    st.markdown(f"#### {title}")
    for key, value in values.items():
        if isinstance(value, (int, float)):
            st.write(f"{key}: {float(value):.2f}")
        else:
            st.write(f"{key}: {value}")


def build_phase1_context(sensor_input: dict, recommendations: dict) -> str:
    lines = [
        "Phase 1 sensor input:",
        f"- N: {sensor_input['N']}",
        f"- P: {sensor_input['P']}",
        f"- K: {sensor_input['K']}",
        f"- ph: {sensor_input['ph']}",
        f"- EC_uS_cm: {sensor_input['EC_uS_cm']}",
        "",
        "Model recommendations:",
    ]
    for key, value in recommendations.items():
        lines.append(f"- {key}: {value}")
    return "\n".join(lines)


def get_llm_response(api_key: str, model_name: str, context_text: str, chat_history: list[dict]) -> str:
    client = OpenAI(
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1",
    )

    system_prompt = (
        "You are a soil analysis expert for paddy rice farming. "
        "Give BRIEF, PRACTICAL answers focused on soil amendments and paddy transplanting actions. "
        "Keep responses to 2-3 short sentences max. Include units. Do NOT invent sensor values. "
        "Be specific to the soil sensor readings and recommended outputs provided."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "system",
            "content": f"Use this context for the current field:\n{context_text}",
        },
    ]
    messages.extend(chat_history)

    completion = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0.2,
        max_tokens=150,
    )
    return completion.choices[0].message.content or "No response generated."


st.title("KrishiLink Soil Analysis Dashboard")
st.caption("Enter sensor values for each phase and get AI-based recommendations.")

with st.sidebar:
    st.header("LLM Settings")
    groq_api_key = os.getenv("GROQ_API_KEY", "")
    llm_model_name = st.text_input(
        "Groq Model",
        value="llama-3.1-8b-instant",
        help="Default is a commonly used Groq free-tier model.",
    )
    if groq_api_key:
        st.success("GROQ_API_KEY loaded from .env")
    else:
        st.error("GROQ_API_KEY not found in .env")

phase1_tab, phase2_tab = st.tabs(["Phase 1 - Dry Soil", "Phase 2 - Muddy Soil"])


with phase1_tab:
    st.subheader("Phase 1 Sensor Inputs")
    st.write("Sensors: N, P, K, pH, EC")

    with st.form("phase1_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            n_value = st.number_input("N", min_value=0.0, max_value=100.0, value=40.0, step=1.0)
            p_value = st.number_input("P", min_value=0.0, max_value=70.0, value=20.0, step=1.0)
        with col2:
            k_value = st.number_input("K", min_value=0.0, max_value=55.0, value=15.0, step=1.0)
            ph_value = st.number_input("pH", min_value=3.5, max_value=9.0, value=6.5, step=0.1)
        with col3:
            ec_value = st.number_input("EC (uS/cm)", min_value=0.0, max_value=3500.0, value=800.0, step=10.0)

        phase1_submit = st.form_submit_button("Get Phase 1 Recommendation")

    if phase1_submit:
        sensor_input = {
            "N": n_value,
            "P": p_value,
            "K": k_value,
            "ph": ph_value,
            "EC_uS_cm": ec_value,
        }
        try:
            phase1_result = phase1_predict(
                N=n_value,
                P=p_value,
                K=k_value,
                ph=ph_value,
                EC=ec_value,
            )

            st.success("Phase 1 recommendation generated.")
            flat_phase1 = flatten_recommendations(phase1_result)

            st.session_state["phase1_sensor_input"] = sensor_input
            st.session_state["phase1_recommendations"] = flat_phase1
            st.session_state["phase1_chat_messages"] = []
            st.session_state["phase1_chat_initialized"] = False

            render_vertical_list("Sensor Data", sensor_input)
            render_vertical_list("Recommended Outputs", flat_phase1)

        except ValueError as error:
            st.error(str(error))

    st.markdown("---")
    st.subheader("Farmer Chat Assistant (Phase 1)")

    if "phase1_sensor_input" in st.session_state and "phase1_recommendations" in st.session_state:
        context_text = build_phase1_context(
            st.session_state["phase1_sensor_input"],
            st.session_state["phase1_recommendations"],
        )

        if not groq_api_key:
            st.info("Add GROQ_API_KEY in .env to start the Phase 1 chat assistant.")
        else:
            if not st.session_state.get("phase1_chat_initialized", False):
                with st.spinner("Generating initial advisor response..."):
                    try:
                        initial_prompt = {
                            "role": "user",
                            "content": (
                                "Based on these soil sensor values and recommendations, what are the top 2-3 actions "
                                "to prepare the soil for paddy transplanting?"
                            ),
                        }
                        initial_response = get_llm_response(
                            groq_api_key,
                            llm_model_name,
                            context_text,
                            [initial_prompt],
                        )
                        st.session_state["phase1_chat_messages"].append(
                            {"role": "assistant", "content": initial_response}
                        )
                        st.session_state["phase1_chat_initialized"] = True
                    except Exception as error:
                        st.error(f"LLM request failed: {error}")

            for message in st.session_state.get("phase1_chat_messages", []):
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            user_prompt = st.chat_input("Ask follow-up questions about Phase 1 recommendations")
            if user_prompt:
                st.session_state["phase1_chat_messages"].append(
                    {"role": "user", "content": user_prompt}
                )

                with st.chat_message("user"):
                    st.markdown(user_prompt)

                with st.spinner("Thinking..."):
                    try:
                        assistant_response = get_llm_response(
                            groq_api_key,
                            llm_model_name,
                            context_text,
                            st.session_state["phase1_chat_messages"],
                        )
                        st.session_state["phase1_chat_messages"].append(
                            {"role": "assistant", "content": assistant_response}
                        )
                    except Exception as error:
                        st.error(f"LLM request failed: {error}")

                if st.session_state.get("phase1_chat_messages"):
                    with st.chat_message("assistant"):
                        st.markdown(st.session_state["phase1_chat_messages"][-1]["content"])
    else:
        st.info("Submit Phase 1 sensor values first to start contextual chat.")


with phase2_tab:
    st.subheader("Phase 2 Sensor Input")
    st.write("Sensor: ORP")

    with st.form("phase2_form"):
        orp_value = st.number_input("ORP (mV)", min_value=-350.0, max_value=350.0, value=-100.0, step=10.0)
        phase2_submit = st.form_submit_button("Get Phase 2 Recommendation")

    if phase2_submit:
        sensor_input = {
            "ORP_mV": orp_value,
        }
        try:
            phase2_result = phase2_predict(ORP=orp_value)

            st.success("Phase 2 recommendation generated.")

            render_vertical_list("Sensor Data", sensor_input)
            render_vertical_list("Recommended Outputs", phase2_result)

        except ValueError as error:
            st.error(str(error))