import streamlit as st
from datetime import datetime, timedelta

st.set_page_config(page_title="GymSlot", page_icon="🏋️", layout="wide")

st.title("GymSlot")
st.subheader("Reserve gym equipment before you arrive")

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    st.header("Login")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        st.session_state.logged_in = True
        st.rerun()

else:
    st.sidebar.title("GymSlot")
    if st.sidebar.button("Log out"):
        st.session_state.logged_in = False
        st.rerun()

    st.header("Book Equipment")

    equipment = st.selectbox(
        "Choose equipment",
        ["Treadmill", "Bench Press", "Squat Rack", "Cable Machine", "Leg Press", "Rowing Machine"]
    )

    date = st.date_input("Choose date")
    time = st.selectbox(
        "Choose time slot",
        ["06:00", "07:00", "08:00", "09:00", "10:00", "17:00", "18:00", "19:00", "20:00"]
    )

    if st.button("Reserve Slot"):
        st.success(f"Booked {equipment} on {date} at {time}")

    st.divider()

    st.header("Your Bookings")
    st.info("Your reserved equipment will appear here.")
