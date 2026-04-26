# file: gymslot_prototype.py

import os
import tkinter as tk
from tkinter import ttk, messagebox
from datetime import datetime, timedelta


MEMBER_ACCOUNTS = {
    "alex@gymslotdemo.com": {"password": "demo123", "name": "Alex Morgan"},
    "sophia@gymslotdemo.com": {"password": "demo123", "name": "Sophia Patel"},
}

STAFF_ACCOUNTS = {
    "STAFF001": {"password": "admin123", "name": "Emma Carter", "role": "Gym Manager"},
    "STAFF002": {"password": "admin123", "name": "Liam Brooks", "role": "Operations Lead"},
}

EQUIPMENT = [
    {"id": "SQ-01", "name": "Squat Rack 1", "zone": "Strength", "type": "Rack", "status": "Available"},
    {"id": "SQ-02", "name": "Squat Rack 2", "zone": "Strength", "type": "Rack", "status": "Booked"},
    {"id": "SQ-03", "name": "Squat Rack 3", "zone": "Strength", "type": "Rack", "status": "Available"},
    {"id": "BN-01", "name": "Bench 1", "zone": "Free Weights", "type": "Bench", "status": "Available"},
    {"id": "BN-02", "name": "Bench 2", "zone": "Free Weights", "type": "Bench", "status": "In Use"},
    {"id": "CB-01", "name": "Cable Machine 1", "zone": "Functional", "type": "Cable", "status": "Available"},
    {"id": "CB-02", "name": "Cable Machine 2", "zone": "Functional", "type": "Cable", "status": "Available"},
    {"id": "TR-01", "name": "Treadmill 1", "zone": "Cardio", "type": "Cardio", "status": "Booked"},
    {"id": "TR-02", "name": "Treadmill 2", "zone": "Cardio", "type": "Cardio", "status": "Available"},
    {"id": "BK-01", "name": "Spin Bike 1", "zone": "Cardio", "type": "Cardio", "status": "Available"},
]

TIME_SLOTS = [
    "06:00 PM - 06:30 PM",
    "06:30 PM - 07:00 PM",
    "07:00 PM - 07:30 PM",
    "07:30 PM - 08:00 PM",
    "08:00 PM - 08:30 PM",
]


class GymSlotApp:
    def __init__(self, root):
        self.root = root
        self.root.title("GymSlot Prototype")
        self.root.geometry("1460x920")
        self.root.minsize(1220, 780)
        self.root.configure(bg="#07111f")

        self.current_user_name = ""
        self.current_staff_name = ""
        self.current_staff_role = ""
        self.current_member_section = "overview"
        self.bookings = []

        self.canvas = None
        self.scrollbar = None
        self.scrollable_frame = None
        self.booking_list_frame = None

        self.member_email_entry = None
        self.member_password_entry = None
        self.staff_id_entry = None
        self.staff_password_entry = None
        self.selected_equipment = None
        self.selected_slot = None
        self.date_entry = None

        self.colors = {
            "bg": "#07111f",
            "bg_alt": "#0b1730",
            "panel": "#0d1c35",
            "panel_2": "#122445",
            "panel_3": "#18345f",
            "panel_4": "#1b3d6e",
            "text": "#f6f8fc",
            "muted": "#9db0ca",
            "accent": "#d8f6f5",
            "accent_dark": "#10233f",
            "accent_blue": "#4f8cff",
            "accent_blue_hover": "#3f79ef",
            "accent_green": "#2dd4bf",
            "input_bg": "#091427",
            "border": "#21385f",
            "success": "#22c55e",
            "warning": "#f59e0b",
            "danger": "#ef4444",
            "white_surface": "#f3f6fb",
        }

        self.fonts = {
            "hero_big": ("Helvetica", 38, "bold"),
            "hero_mid": ("Helvetica", 25, "bold"),
            "title": ("Helvetica", 20, "bold"),
            "subtitle": ("Helvetica", 12),
            "card_title": ("Helvetica", 14, "bold"),
            "body": ("Helvetica", 11),
            "small": ("Helvetica", 10),
            "metric_value": ("Helvetica", 24, "bold"),
            "metric_label": ("Helvetica", 10, "bold"),
            "button": ("Helvetica", 11, "bold"),
            "nav_brand": ("Helvetica", 16, "bold"),
            "sidebar_title": ("Helvetica", 12, "bold"),
        }

        self.logo_original = None
        self.logo_home = None
        self.logo_small = None
        self.logo_tiny = None
        self.load_logo_assets()

        self.style = ttk.Style()
        try:
            self.style.theme_use("clam")
        except tk.TclError:
            pass
        self.configure_ttk_styles()

        self.page_host = tk.Frame(self.root, bg=self.colors["bg"])
        self.page_host.pack(fill="both", expand=True)

        self.show_landing_page()

    def load_logo_assets(self):
        logo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gymslot_logo.png")
        if not os.path.exists(logo_path):
            return

        try:
            self.logo_original = tk.PhotoImage(file=logo_path)
        except tk.TclError:
            self.logo_original = None
            return

        self.logo_home = self.create_scaled_logo(320)
        self.logo_small = self.create_scaled_logo(150)
        self.logo_tiny = self.create_scaled_logo(110)

    def create_scaled_logo(self, target_width):
        if self.logo_original is None:
            return None
        width = self.logo_original.width()
        factor = max(1, width // target_width)
        return self.logo_original.subsample(factor, factor)

    def configure_ttk_styles(self):
        self.style.configure(
            "Dark.TCombobox",
            fieldbackground=self.colors["input_bg"],
            background=self.colors["input_bg"],
            foreground=self.colors["text"],
            bordercolor=self.colors["border"],
            arrowcolor=self.colors["text"],
            padding=8,
            relief="flat",
        )
        self.style.map(
            "Dark.TCombobox",
            fieldbackground=[("readonly", self.colors["input_bg"])],
            background=[("readonly", self.colors["input_bg"])],
            foreground=[("readonly", self.colors["text"])],
            selectforeground=[("readonly", self.colors["text"])],
            selectbackground=[("readonly", self.colors["input_bg"])],
        )

    def clear_page(self):
        for widget in self.page_host.winfo_children():
            widget.destroy()

    def create_scrollable_page(self):
        self.clear_page()

        outer = tk.Frame(self.page_host, bg=self.colors["bg"])
        outer.pack(fill="both", expand=True)

        self.canvas = tk.Canvas(
            outer,
            bg=self.colors["bg"],
            highlightthickness=0,
            bd=0,
        )
        self.scrollbar = tk.Scrollbar(
            outer,
            orient="vertical",
            command=self.canvas.yview,
            troughcolor=self.colors["bg"],
            bg=self.colors["panel_3"],
            activebackground=self.colors["panel_3"],
        )

        self.scrollable_frame = tk.Frame(self.canvas, bg=self.colors["bg"])

        self.scrollable_frame.bind(
            "<Configure>",
            lambda event: self.canvas.configure(scrollregion=self.canvas.bbox("all")),
        )

        window_id = self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")

        def resize_scrollable_frame(event):
            self.canvas.itemconfig(window_id, width=event.width)

        self.canvas.bind("<Configure>", resize_scrollable_frame)
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

        self.bind_mousewheel_recursive(self.canvas)
        self.bind_mousewheel_recursive(self.scrollable_frame)

        return self.scrollable_frame

    def on_mousewheel(self, event):
        if self.canvas is None:
            return
        if hasattr(event, "delta") and event.delta:
            self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        elif getattr(event, "num", None) == 4:
            self.canvas.yview_scroll(-3, "units")
        elif getattr(event, "num", None) == 5:
            self.canvas.yview_scroll(3, "units")

    def bind_mousewheel_recursive(self, widget):
        try:
            widget.bind("<MouseWheel>", self.on_mousewheel)
            widget.bind("<Button-4>", self.on_mousewheel)
            widget.bind("<Button-5>", self.on_mousewheel)
        except tk.TclError:
            pass

        for child in widget.winfo_children():
            self.bind_mousewheel_recursive(child)

    def scroll_to_top(self):
        if self.canvas is not None:
            self.canvas.yview_moveto(0)

    def create_card(self, parent, bg=None, padx=18, pady=18):
        return tk.Frame(
            parent,
            bg=bg or self.colors["panel"],
            highlightthickness=1,
            highlightbackground=self.colors["border"],
            padx=padx,
            pady=pady,
        )

    def create_label(self, parent, text, font_key="body", color=None, bg=None, **kwargs):
        return tk.Label(
            parent,
            text=text,
            font=self.fonts[font_key],
            fg=color or self.colors["text"],
            bg=bg or parent.cget("bg"),
            **kwargs,
        )

    def create_section_title(self, parent, title, subtitle):
        self.create_label(parent, title, "title").pack(anchor="w")
        self.create_label(
            parent,
            subtitle,
            "subtitle",
            self.colors["muted"],
            anchor="w",
            justify="left",
            wraplength=1000,
        ).pack(anchor="w", pady=(6, 0))

    def create_input(self, parent, label_text, show=""):
        self.create_label(parent, label_text, "body").pack(anchor="w", pady=(0, 6))
        entry = tk.Entry(
            parent,
            font=self.fonts["body"],
            fg=self.colors["text"],
            bg=self.colors["input_bg"],
            insertbackground=self.colors["text"],
            relief="flat",
            highlightthickness=1,
            highlightbackground=self.colors["border"],
            highlightcolor=self.colors["accent_blue"],
            bd=0,
            show=show,
        )
        entry.pack(fill="x", ipady=12, pady=(0, 16))
        return entry

    def create_action_button(self, parent, text, command, bg=None, hover_bg=None, fg="white"):
        normal_bg = bg or self.colors["accent_blue"]
        hover_color = hover_bg or self.colors["accent_blue_hover"]

        button = tk.Frame(
            parent,
            bg=normal_bg,
            highlightthickness=1,
            highlightbackground=normal_bg,
            cursor="hand2",
        )

        label = tk.Label(
            button,
            text=text,
            font=self.fonts["button"],
            fg=fg,
            bg=normal_bg,
            padx=16,
            pady=11,
            cursor="hand2",
        )
        label.pack(fill="both", expand=True)

        def on_enter(_event):
            button.configure(bg=hover_color, highlightbackground=hover_color)
            label.configure(bg=hover_color)

        def on_leave(_event):
            button.configure(bg=normal_bg, highlightbackground=normal_bg)
            label.configure(bg=normal_bg)

        def on_click(_event):
            command()

        for target in (button, label):
            target.bind("<Enter>", on_enter)
            target.bind("<Leave>", on_leave)
            target.bind("<Button-1>", on_click)

        return button

    def create_secondary_button(self, parent, text, command):
        return self.create_action_button(
            parent,
            text,
            command,
            bg=self.colors["panel_3"],
            hover_bg=self.colors["panel_4"],
            fg="white",
        )

    def create_outline_button(self, parent, text, command):
        button = tk.Frame(
            parent,
            bg=self.colors["bg"],
            highlightthickness=1,
            highlightbackground=self.colors["border"],
            cursor="hand2",
        )

        label = tk.Label(
            button,
            text=text,
            font=self.fonts["button"],
            fg=self.colors["text"],
            bg=self.colors["bg"],
            padx=16,
            pady=10,
            cursor="hand2",
        )
        label.pack(fill="both", expand=True)

        def on_enter(_event):
            button.configure(bg=self.colors["panel_2"])
            label.configure(bg=self.colors["panel_2"])

        def on_leave(_event):
            button.configure(bg=self.colors["bg"])
            label.configure(bg=self.colors["bg"])

        def on_click(_event):
            command()

        for target in (button, label):
            target.bind("<Enter>", on_enter)
            target.bind("<Leave>", on_leave)
            target.bind("<Button-1>", on_click)

        return button

    def create_metric_card(self, parent, title, value, note=""):
        card = self.create_card(parent, bg=self.colors["panel_2"], padx=18, pady=18)
        card.pack(side="left", fill="both", expand=True, padx=7)
        self.create_label(card, title, "metric_label", self.colors["muted"]).pack(anchor="w")
        self.create_label(card, value, "metric_value").pack(anchor="w", pady=(8, 4))
        if note:
            self.create_label(card, note, "small", self.colors["accent_blue"]).pack(anchor="w")

    def create_sidebar_button(self, parent, text, section_name):
        is_active = self.current_member_section == section_name
        bg = self.colors["accent_blue"] if is_active else self.colors["panel"]
        fg = "#ffffff" if is_active else self.colors["text"]
        hover = self.colors["accent_blue_hover"] if is_active else self.colors["panel_2"]

        button = tk.Frame(
            parent,
            bg=bg,
            highlightthickness=1,
            highlightbackground=self.colors["border"] if not is_active else bg,
            cursor="hand2",
        )

        label = tk.Label(
            button,
            text=text,
            font=self.fonts["button"],
            fg=fg,
            bg=bg,
            padx=12,
            pady=10,
            anchor="w",
            cursor="hand2",
        )
        label.pack(fill="x")

        def on_enter(_event):
            button.configure(bg=hover)
            label.configure(bg=hover)

        def on_leave(_event):
            button.configure(bg=bg)
            label.configure(bg=bg)

        def on_click(_event):
            self.show_member_dashboard(section_name)

        for target in (button, label):
            target.bind("<Enter>", on_enter)
            target.bind("<Leave>", on_leave)
            target.bind("<Button-1>", on_click)

        return button

    def create_home_topbar(self, parent):
        nav = tk.Frame(parent, bg=self.colors["bg"])
        nav.pack(fill="x", padx=28, pady=(20, 18))

        left = tk.Frame(nav, bg=self.colors["bg"])
        left.pack(side="left", fill="x", expand=True)

        if self.logo_tiny:
            tk.Label(left, image=self.logo_tiny, bg=self.colors["bg"]).pack(side="left", padx=(0, 10))
        else:
            dot = tk.Canvas(left, width=16, height=16, bg=self.colors["bg"], highlightthickness=0)
            dot.create_oval(2, 2, 14, 14, fill=self.colors["accent_green"], outline=self.colors["accent_green"])
            dot.pack(side="left", padx=(0, 8))
            tk.Label(
                left,
                text="GymSlot",
                font=self.fonts["nav_brand"],
                fg=self.colors["text"],
                bg=self.colors["bg"],
            ).pack(side="left")

        right = tk.Frame(nav, bg=self.colors["bg"])
        right.pack(side="right")

        self.create_outline_button(right, "Staff Demo", self.focus_staff_login).pack(side="right")
        self.create_action_button(
            right,
            "Join Demo",
            self.focus_member_login,
            bg=self.colors["accent"],
            hover_bg="#c8efef",
            fg=self.colors["accent_dark"],
        ).pack(side="right", padx=(0, 10))

    def create_logged_in_navbar(self, parent, title, subtitle):
        nav = tk.Frame(parent, bg=self.colors["bg"])
        nav.pack(fill="x", padx=28, pady=(20, 16))

        left = tk.Frame(nav, bg=self.colors["bg"])
        left.pack(side="left", fill="x", expand=True)

        brand_row = tk.Frame(left, bg=self.colors["bg"])
        brand_row.pack(anchor="w", pady=(0, 10))

        if self.logo_tiny:
            tk.Label(brand_row, image=self.logo_tiny, bg=self.colors["bg"]).pack(side="left", padx=(0, 10))
        else:
            dot = tk.Canvas(brand_row, width=14, height=14, bg=self.colors["bg"], highlightthickness=0)
            dot.create_oval(2, 2, 12, 12, fill=self.colors["accent_green"], outline=self.colors["accent_green"])
            dot.pack(side="left", padx=(0, 8))
            tk.Label(
                brand_row,
                text="GymSlot",
                font=self.fonts["nav_brand"],
                fg=self.colors["text"],
                bg=self.colors["bg"],
            ).pack(side="left")

        self.create_label(left, title, "hero_mid").pack(anchor="w")
        self.create_label(
            left,
            subtitle,
            "subtitle",
            self.colors["muted"],
            anchor="w",
            justify="left",
            wraplength=900,
        ).pack(anchor="w", pady=(6, 0))

        right = tk.Frame(nav, bg=self.colors["bg"])
        right.pack(side="right")

        self.create_secondary_button(right, "Scroll to Top", self.scroll_to_top).pack(side="right")
        self.create_secondary_button(right, "Back to Login", self.show_landing_page).pack(side="right", padx=(0, 10))

    def focus_member_login(self):
        try:
            self.member_email_entry.focus_set()
        except AttributeError:
            pass

    def focus_staff_login(self):
        try:
            self.staff_id_entry.focus_set()
        except AttributeError:
            pass

    def show_landing_page(self):
        page = self.create_scrollable_page()

        self.create_home_topbar(page)

        hero = tk.Frame(page, bg=self.colors["bg"])
        hero.pack(fill="x", padx=28, pady=(4, 18))

        hero_card = tk.Frame(
            hero,
            bg=self.colors["bg_alt"],
            highlightthickness=1,
            highlightbackground=self.colors["border"],
            padx=30,
            pady=34,
        )
        hero_card.pack(fill="x")

        if self.logo_home:
            logo_wrap = tk.Frame(hero_card, bg=self.colors["bg_alt"])
            logo_wrap.pack(fill="x", pady=(0, 16))
            tk.Label(logo_wrap, image=self.logo_home, bg=self.colors["bg_alt"]).pack()

        tk.Label(
            hero_card,
            text="Welcome to Feeling\nGymSlot Good",
            font=self.fonts["hero_big"],
            fg=self.colors["text"],
            bg=self.colors["bg_alt"],
            justify="center",
        ).pack(fill="x", pady=(0, 8))

        tk.Label(
            hero_card,
            text="Reserve gym equipment in advance. Arrive with a plan. Train with less waiting.",
            font=("Helvetica", 14, "bold"),
            fg=self.colors["text"],
            bg=self.colors["bg_alt"],
            justify="center",
        ).pack(fill="x", pady=(0, 8))

        tk.Label(
            hero_card,
            text="Built for members and designed for gyms that want a smoother peak-time experience.",
            font=self.fonts["subtitle"],
            fg=self.colors["muted"],
            bg=self.colors["bg_alt"],
            justify="center",
        ).pack(fill="x")

        search_bar = tk.Frame(
            hero_card,
            bg=self.colors["white_surface"],
            highlightthickness=1,
            highlightbackground="#dfe7f0",
            padx=16,
            pady=12,
        )
        search_bar.pack(pady=(28, 12), ipadx=12)

        tk.Label(
            search_bar,
            text="🔎",
            font=("Helvetica", 13),
            bg=self.colors["white_surface"],
            fg="#617285",
        ).pack(side="left", padx=(0, 10))

        tk.Label(
            search_bar,
            text="Search by equipment, zone, or time slot",
            font=("Helvetica", 12),
            bg=self.colors["white_surface"],
            fg="#7a8897",
        ).pack(side="left")

        ctas = tk.Frame(hero_card, bg=self.colors["bg_alt"])
        ctas.pack(pady=(18, 2))

        self.create_action_button(
            ctas,
            "Member Login",
            self.focus_member_login,
            bg=self.colors["accent"],
            hover_bg="#c8efef",
            fg=self.colors["accent_dark"],
        ).pack(side="left", padx=6)
        self.create_secondary_button(ctas, "Staff Login", self.focus_staff_login).pack(side="left", padx=6)

        why_section = self.create_card(page, bg=self.colors["panel"], padx=24, pady=24)
        why_section.pack(fill="x", padx=28, pady=(0, 18))

        tk.Label(
            why_section,
            text="Why GymSlot?",
            font=self.fonts["hero_mid"],
            fg=self.colors["text"],
            bg=self.colors["panel"],
        ).pack(anchor="center", pady=(0, 6))
        tk.Label(
            why_section,
            text="A more organised gym experience for members and better visibility for operators.",
            font=self.fonts["subtitle"],
            fg=self.colors["muted"],
            bg=self.colors["panel"],
        ).pack(anchor="center")

        feature_row = tk.Frame(why_section, bg=self.colors["panel"])
        feature_row.pack(fill="x", pady=(22, 0))

        features = [
            (
                "Reserve Ahead",
                "Members can secure key equipment like squat racks, benches, cable stations, and cardio machines before peak hours.",
            ),
            (
                "Reduce Friction",
                "The platform helps gyms reduce uncertainty and waiting, especially during busy evening sessions.",
            ),
            (
                "Use Real Data",
                "Management can analyse utilisation, no-shows, and demand pressure to support layout and investment decisions.",
            ),
        ]

        for title, body in features:
            card = self.create_card(feature_row, bg=self.colors["panel_2"], padx=18, pady=18)
            card.pack(side="left", fill="both", expand=True, padx=7)
            self.create_label(card, title, "card_title").pack(anchor="w")
            self.create_label(
                card,
                body,
                "body",
                self.colors["muted"],
                wraplength=320,
                justify="left",
                anchor="w",
            ).pack(fill="x", pady=(10, 0))

        demo_card = self.create_card(page, bg=self.colors["panel_2"], padx=20, pady=18)
        demo_card.pack(fill="x", padx=28, pady=(0, 18))
        self.create_label(
            demo_card,
            "Demo Member: alex@gymslotdemo.com / demo123\nDemo Staff: STAFF001 / admin123",
            "body",
            self.colors["text"],
            anchor="w",
            justify="left",
        ).pack(fill="x")

        login_wrap = tk.Frame(page, bg=self.colors["bg"])
        login_wrap.pack(fill="both", expand=True, padx=28, pady=(0, 28))

        member_card = self.create_card(login_wrap, bg=self.colors["panel"], padx=24, pady=24)
        member_card.pack(side="left", fill="both", expand=True, padx=(0, 10))

        staff_card = self.create_card(login_wrap, bg=self.colors["panel"], padx=24, pady=24)
        staff_card.pack(side="left", fill="both", expand=True, padx=(10, 0))

        self.create_section_title(
            member_card,
            "Member Login",
            "Reserve equipment, see live availability, and manage bookings.",
        )
        tk.Frame(member_card, bg=self.colors["panel"], height=16).pack()
        self.member_email_entry = self.create_input(member_card, "Email")
        self.member_password_entry = self.create_input(member_card, "Password", show="*")
        self.create_action_button(member_card, "Login as Member", self.handle_member_login).pack(fill="x", pady=(4, 0))

        self.create_section_title(
            staff_card,
            "Staff Login",
            "Access utilisation, demand, and no-show analytics for your gym.",
        )
        tk.Frame(staff_card, bg=self.colors["panel"], height=16).pack()
        self.staff_id_entry = self.create_input(staff_card, "Staff ID")
        self.staff_password_entry = self.create_input(staff_card, "Password", show="*")
        self.create_action_button(staff_card, "Login as Staff", self.handle_staff_login).pack(fill="x", pady=(4, 0))

        self.bind_mousewheel_recursive(page)
        self.scroll_to_top()

    def handle_member_login(self):
        email = self.member_email_entry.get().strip().lower()
        password = self.member_password_entry.get().strip()
        record = MEMBER_ACCOUNTS.get(email)

        if not record or record["password"] != password:
            messagebox.showerror("Login Failed", "Invalid member email or password.")
            return

        self.current_user_name = record["name"]
        self.show_member_dashboard("overview")

    def handle_staff_login(self):
        staff_id = self.staff_id_entry.get().strip().upper()
        password = self.staff_password_entry.get().strip()
        record = STAFF_ACCOUNTS.get(staff_id)

        if not record or record["password"] != password:
            messagebox.showerror("Login Failed", "Invalid staff ID or password.")
            return

        self.current_staff_name = record["name"]
        self.current_staff_role = record["role"]
        self.show_staff_dashboard()

    def show_member_dashboard(self, section="overview"):
        self.current_member_section = section
        page = self.create_scrollable_page()

        self.create_logged_in_navbar(
            page,
            f"Welcome back, {self.current_user_name}",
            "Use the sidebar to move between booking, live availability, your bookings, and QR check-in.",
        )

        metrics = tk.Frame(page, bg=self.colors["bg"])
        metrics.pack(fill="x", padx=28, pady=(0, 18))
        self.create_metric_card(metrics, "Live Availability", "67%")
        self.create_metric_card(metrics, "Peak Window", "6–8 PM")
        self.create_metric_card(metrics, "Your Bookings", str(len(self.bookings)))
        self.create_metric_card(metrics, "Check-In Rule", "5 min")

        body = tk.Frame(page, bg=self.colors["bg"])
        body.pack(fill="both", expand=True, padx=28, pady=(0, 28))

        sidebar = self.create_card(body, bg=self.colors["panel"], padx=18, pady=18)
        sidebar.pack(side="left", fill="y", padx=(0, 12))

        content = tk.Frame(body, bg=self.colors["bg"])
        content.pack(side="left", fill="both", expand=True)

        self.build_member_sidebar(sidebar)

        if section == "overview":
            self.build_member_overview(content)
        elif section == "floor":
            self.build_member_floor_map(content)
        elif section == "reserve":
            self.build_member_reserve(content)
        elif section == "bookings":
            self.build_member_bookings(content)
        elif section == "qr":
            self.build_member_qr(content)
        else:
            self.build_member_overview(content)

        self.bind_mousewheel_recursive(page)
        self.scroll_to_top()

    def build_member_sidebar(self, parent):
        if self.logo_small:
            logo_holder = tk.Frame(parent, bg=self.colors["panel"])
            logo_holder.pack(fill="x", pady=(0, 16))
            tk.Label(logo_holder, image=self.logo_small, bg=self.colors["panel"]).pack(anchor="center")

        self.create_label(parent, self.current_user_name, "sidebar_title").pack(anchor="w")
        self.create_label(
            parent,
            "Member Dashboard",
            "small",
            self.colors["muted"],
        ).pack(anchor="w", pady=(4, 16))

        buttons = [
            ("Overview", "overview"),
            ("Live Floor Map", "floor"),
            ("Reserve Slot", "reserve"),
            ("My Bookings", "bookings"),
            ("QR Check-In", "qr"),
        ]

        for label, key in buttons:
            self.create_sidebar_button(parent, label, key).pack(fill="x", pady=(0, 10))

        tk.Frame(parent, bg=self.colors["panel"], height=12).pack()
        self.create_secondary_button(parent, "Back to Login", self.show_landing_page).pack(fill="x")

    def build_member_overview(self, parent):
        hero = self.create_card(parent, bg=self.colors["bg_alt"], padx=22, pady=22)
        hero.pack(fill="x", pady=(0, 16))
        self.create_label(hero, "Plan your workout before you arrive", "hero_mid").pack(anchor="w")
        self.create_label(
            hero,
            "Check live availability, reserve your equipment, and move straight into QR check-in after you book.",
            "subtitle",
            self.colors["muted"],
            anchor="w",
            justify="left",
        ).pack(anchor="w", pady=(8, 0))

        top = tk.Frame(parent, bg=self.colors["bg"])
        top.pack(fill="x", pady=(0, 16))

        left = self.create_card(top, bg=self.colors["panel"], padx=22, pady=22)
        left.pack(side="left", fill="both", expand=True, padx=(0, 8))

        right = self.create_card(top, bg=self.colors["panel"], padx=22, pady=22)
        right.pack(side="left", fill="both", expand=True, padx=(8, 0))

        self.create_section_title(left, "How it works", "A simple member flow designed for convenience.")
        tk.Frame(left, bg=self.colors["panel"], height=14).pack()
        steps = [
            "1. Open Live Floor Map to view equipment status.",
            "2. Go to Reserve Slot to choose your machine and time.",
            "3. After booking, use QR Check-In to confirm your session.",
            "4. View all reservations in My Bookings.",
        ]
        for step in steps:
            self.create_label(left, step, "body", self.colors["muted"], anchor="w", justify="left").pack(anchor="w", pady=4)

        self.create_section_title(right, "Quick actions", "Jump straight to the page you need.")
        tk.Frame(right, bg=self.colors["panel"], height=14).pack()
        actions = tk.Frame(right, bg=self.colors["panel"])
        actions.pack(anchor="w")
        self.create_secondary_button(actions, "Go to Floor Map", lambda: self.show_member_dashboard("floor")).pack(side="left", padx=(0, 10))
        self.create_secondary_button(actions, "Reserve Now", lambda: self.show_member_dashboard("reserve")).pack(side="left", padx=(0, 10))
        self.create_action_button(actions, "My Bookings", lambda: self.show_member_dashboard("bookings")).pack(side="left")

        bottom = self.create_card(parent, bg=self.colors["panel"], padx=22, pady=22)
        bottom.pack(fill="x")

        self.create_section_title(bottom, "Popular right now", "A quick member-facing snapshot of high-demand equipment.")
        tk.Frame(bottom, bg=self.colors["panel"], height=16).pack()

        highlights = [
            ("Squat Racks", "High demand during 18:00–20:00", self.colors["warning"]),
            ("Benches", "Strong evening demand", self.colors["warning"]),
            ("Cable Machines", "Steady all-day usage", self.colors["success"]),
            ("Cardio", "Moderate demand right now", self.colors["success"]),
        ]
        grid = tk.Frame(bottom, bg=self.colors["panel"])
        grid.pack(fill="x")
        for title, note, color in highlights:
            card = self.create_card(grid, bg=self.colors["panel_2"], padx=16, pady=16)
            card.pack(side="left", fill="both", expand=True, padx=6)
            self.create_label(card, title, "card_title").pack(anchor="w")
            self.create_label(card, note, "small", color).pack(anchor="w", pady=(6, 0))

    def build_member_floor_map(self, parent):
        panel = self.create_card(parent, bg=self.colors["panel"], padx=22, pady=22)
        panel.pack(fill="both", expand=True)

        self.create_section_title(
            panel,
            "Live Floor Map",
            "A sectioned view of equipment availability across the gym floor.",
        )
        tk.Frame(panel, bg=self.colors["panel"], height=18).pack()

        grid = tk.Frame(panel, bg=self.colors["panel"])
        grid.pack(fill="both", expand=True)

        for col in range(3):
            grid.grid_columnconfigure(col, weight=1)

        for index, item in enumerate(EQUIPMENT):
            row = index // 3
            col = index % 3

            card = self.create_card(grid, bg=self.colors["panel_2"], padx=16, pady=16)
            card.grid(row=row, column=col, sticky="nsew", padx=8, pady=8)

            zone_chip = tk.Label(
                card,
                text=item["zone"],
                font=self.fonts["small"],
                fg="#dceafe",
                bg=self.colors["panel_3"],
                padx=10,
                pady=5,
            )
            zone_chip.pack(anchor="w")

            self.create_label(card, item["name"], "card_title").pack(anchor="w", pady=(12, 6))
            self.create_label(card, f"ID: {item['id']}   Type: {item['type']}", "body", self.colors["muted"]).pack(anchor="w")
            self.create_label(card, f"● {item['status']}", "body", self.get_status_color(item["status"])).pack(anchor="w", pady=(12, 0))

    def build_member_reserve(self, parent):
        wrap = tk.Frame(parent, bg=self.colors["bg"])
        wrap.pack(fill="both", expand=True)

        left = self.create_card(wrap, bg=self.colors["panel"], padx=22, pady=22)
        left.pack(side="left", fill="both", expand=True, padx=(0, 8))

        right = self.create_card(wrap, bg=self.colors["panel"], padx=22, pady=22)
        right.pack(side="left", fill="both", expand=True, padx=(8, 0))

        self.create_section_title(left, "Reserve Equipment", "Choose your machine and a 30-minute slot.")
        tk.Frame(left, bg=self.colors["panel"], height=18).pack()

        self.create_label(left, "Equipment", "body").pack(anchor="w", pady=(0, 6))
        equipment_names = [item["name"] for item in EQUIPMENT]
        self.selected_equipment = tk.StringVar(value=equipment_names[0])

        equipment_box = ttk.Combobox(
            left,
            textvariable=self.selected_equipment,
            values=equipment_names,
            state="readonly",
            style="Dark.TCombobox",
        )
        equipment_box.pack(fill="x", pady=(0, 16), ipady=4)

        self.create_label(left, "Date", "body").pack(anchor="w", pady=(0, 6))
        self.date_entry = tk.Entry(
            left,
            font=self.fonts["body"],
            fg=self.colors["text"],
            bg=self.colors["input_bg"],
            insertbackground=self.colors["text"],
            relief="flat",
            highlightthickness=1,
            highlightbackground=self.colors["border"],
            highlightcolor=self.colors["accent_blue"],
            bd=0,
        )
        self.date_entry.insert(0, (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d"))
        self.date_entry.pack(fill="x", ipady=12, pady=(0, 16))

        self.create_label(left, "Time Slot", "body").pack(anchor="w", pady=(0, 6))
        self.selected_slot = tk.StringVar(value=TIME_SLOTS[0])

        slot_box = ttk.Combobox(
            left,
            textvariable=self.selected_slot,
            values=TIME_SLOTS,
            state="readonly",
            style="Dark.TCombobox",
        )
        slot_box.pack(fill="x", pady=(0, 16), ipady=4)

        self.create_action_button(left, "Reserve Slot", self.handle_reservation).pack(fill="x", pady=(2, 14))

        self.create_label(
            left,
            "After booking, you will be taken to a QR check-in screen for the pitch demo.",
            "small",
            self.colors["muted"],
            wraplength=420,
            justify="left",
            anchor="w",
        ).pack(fill="x")

        self.create_section_title(right, "Booking tips", "Helpful guidance for a smoother booking experience.")
        tk.Frame(right, bg=self.colors["panel"], height=18).pack()

        tips = [
            ("Peak Time", "Racks and benches are busiest on weekday evenings."),
            ("Best Use", "Reserve equipment for your first working sets to reduce waiting."),
            ("Cardio", "Cardio stations usually have more flexibility."),
            ("Check-In", "Arrive on time to keep your slot active in the full product vision."),
        ]
        for title, text in tips:
            tip = self.create_card(right, bg=self.colors["panel_2"], padx=16, pady=16)
            tip.pack(fill="x", pady=(0, 10))
            self.create_label(tip, title, "card_title").pack(anchor="w")
            self.create_label(
                tip,
                text,
                "body",
                self.colors["muted"],
                wraplength=420,
                justify="left",
                anchor="w",
            ).pack(fill="x", pady=(8, 0))

    def build_member_bookings(self, parent):
        panel = self.create_card(parent, bg=self.colors["panel"], padx=22, pady=22)
        panel.pack(fill="both", expand=True)

        self.create_section_title(
            panel,
            "My Bookings",
            "All reservations created during this demo session.",
        )
        tk.Frame(panel, bg=self.colors["panel"], height=18).pack()

        self.booking_list_frame = tk.Frame(panel, bg=self.colors["panel"])
        self.booking_list_frame.pack(fill="both", expand=True)
        self.refresh_booking_list()

    def build_member_qr(self, parent):
        panel = self.create_card(parent, bg=self.colors["panel"], padx=22, pady=22)
        panel.pack(fill="both", expand=True)

        self.create_section_title(
            panel,
            "QR Check-In",
            "This simulates the check-in flow after a member has reserved equipment.",
        )
        tk.Frame(panel, bg=self.colors["panel"], height=18).pack()

        if not self.bookings:
            empty = self.create_card(panel, bg=self.colors["panel_2"], padx=18, pady=18)
            empty.pack(fill="x")
            self.create_label(empty, "No reservation yet", "card_title").pack(anchor="w")
            self.create_label(
                empty,
                "Make a booking first, then return here to see the fake QR check-in screen.",
                "body",
                self.colors["muted"],
                wraplength=700,
                justify="left",
                anchor="w",
            ).pack(anchor="w", pady=(8, 12))
            self.create_action_button(empty, "Go to Reserve Slot", lambda: self.show_member_dashboard("reserve")).pack(anchor="w")
            return

        latest_booking = self.bookings[-1]

        wrap = tk.Frame(panel, bg=self.colors["panel"])
        wrap.pack(fill="both", expand=True)

        left = self.create_card(wrap, bg=self.colors["panel_2"], padx=20, pady=20)
        left.pack(side="left", fill="both", expand=True, padx=(0, 8))

        right = self.create_card(wrap, bg=self.colors["panel_2"], padx=20, pady=20)
        right.pack(side="left", fill="both", expand=True, padx=(8, 0))

        self.create_label(left, "Booking Summary", "card_title").pack(anchor="w")
        self.create_label(left, f"Equipment: {latest_booking['equipment']}", "body", self.colors["muted"]).pack(anchor="w", pady=(12, 4))
        self.create_label(left, f"Date: {latest_booking['date']}", "body", self.colors["muted"]).pack(anchor="w", pady=4)
        self.create_label(left, f"Time: {latest_booking['slot']}", "body", self.colors["muted"]).pack(anchor="w", pady=4)
        self.create_label(left, f"Status: {latest_booking['status']}", "body", self.get_status_color_from_booking(latest_booking)).pack(anchor="w", pady=(8, 4))
        self.create_label(left, f"Check-In Code: {latest_booking['checkin_code']}", "body", self.colors["accent_blue"]).pack(anchor="w", pady=(8, 0))

        action_row = tk.Frame(left, bg=self.colors["panel_2"])
        action_row.pack(anchor="w", pady=(18, 0))
        self.create_action_button(action_row, "Simulate QR Check-In", self.simulate_checkin).pack(side="left", padx=(0, 10))
        self.create_secondary_button(action_row, "View My Bookings", lambda: self.show_member_dashboard("bookings")).pack(side="left")

        if latest_booking["status"] == "Checked In":
            success = self.create_card(left, bg=self.colors["panel"], padx=14, pady=14)
            success.pack(fill="x", pady=(18, 0))
            self.create_label(success, "Check-in successful", "card_title", self.colors["success"]).pack(anchor="w")
            checked_in_text = latest_booking.get("checked_in_at", "Just now")
            self.create_label(
                success,
                f"The member has been checked in successfully at {checked_in_text}.",
                "body",
                self.colors["muted"],
                wraplength=420,
                justify="left",
                anchor="w",
            ).pack(anchor="w", pady=(8, 0))

        self.create_label(right, "Scan to confirm check-in", "card_title").pack(anchor="center", pady=(0, 14))

        qr_holder = tk.Frame(right, bg="#ffffff", padx=18, pady=18)
        qr_holder.pack(pady=(0, 14))

        qr_canvas = tk.Canvas(
            qr_holder,
            width=250,
            height=250,
            bg="#ffffff",
            highlightthickness=0,
            bd=0,
        )
        qr_canvas.pack()
        self.draw_fake_qr(qr_canvas, latest_booking["checkin_code"])

        self.create_label(
            right,
            "Prototype demo: this QR screen shows what members would use to confirm their reservation at the equipment.",
            "body",
            self.colors["muted"],
            justify="center",
            wraplength=420,
        ).pack(pady=(8, 0))

    def get_status_color(self, status):
        if status == "Available":
            return self.colors["success"]
        if status == "Booked":
            return self.colors["warning"]
        return self.colors["danger"]

    def get_status_color_from_booking(self, booking):
        if booking["status"] == "Checked In":
            return self.colors["success"]
        return self.colors["warning"]

    def draw_fake_qr(self, canvas, code):
        canvas.delete("all")
        size = 250
        grid_size = 25
        cell = size // grid_size

        for row in range(grid_size):
            for col in range(grid_size):
                seed = sum(ord(ch) for ch in code)
                fill = ((row * 7 + col * 11 + seed) % 5 in (0, 2))
                if fill:
                    x1 = col * cell
                    y1 = row * cell
                    x2 = x1 + cell
                    y2 = y1 + cell
                    canvas.create_rectangle(x1, y1, x2, y2, fill="black", outline="black")

        self.draw_finder(canvas, 0, 0, cell)
        self.draw_finder(canvas, (grid_size - 7) * cell, 0, cell)
        self.draw_finder(canvas, 0, (grid_size - 7) * cell, cell)

    def draw_finder(self, canvas, x, y, cell):
        canvas.create_rectangle(x, y, x + 7 * cell, y + 7 * cell, fill="black", outline="black")
        canvas.create_rectangle(x + cell, y + cell, x + 6 * cell, y + 6 * cell, fill="white", outline="white")
        canvas.create_rectangle(x + 2 * cell, y + 2 * cell, x + 5 * cell, y + 5 * cell, fill="black", outline="black")

    def refresh_booking_list(self):
        if self.booking_list_frame is None:
            return

        for widget in self.booking_list_frame.winfo_children():
            widget.destroy()

        if not self.bookings:
            self.create_label(
                self.booking_list_frame,
                "No bookings yet. Create your first reservation in the Reserve Slot section.",
                "body",
                self.colors["muted"],
                anchor="w",
                justify="left",
            ).pack(fill="x")
            return

        for index, booking in enumerate(self.bookings, start=1):
            card = self.create_card(self.booking_list_frame, bg=self.colors["panel_2"], padx=16, pady=16)
            card.pack(fill="x", pady=(0, 10))

            chip = tk.Label(
                card,
                text=f"Booking #{index}",
                font=self.fonts["small"],
                fg="#dceafe",
                bg=self.colors["panel_3"],
                padx=10,
                pady=5,
            )
            chip.pack(anchor="w")

            self.create_label(card, booking["equipment"], "card_title").pack(anchor="w", pady=(12, 6))
            self.create_label(card, booking["date"], "body", self.colors["muted"]).pack(anchor="w")
            self.create_label(card, booking["slot"], "body", self.colors["muted"]).pack(anchor="w", pady=(4, 0))
            self.create_label(card, f"Status: {booking['status']}", "small", self.get_status_color_from_booking(booking)).pack(anchor="w", pady=(10, 0))
            self.create_label(card, f"Code: {booking['checkin_code']}", "small", self.colors["accent_blue"]).pack(anchor="w", pady=(4, 0))

    def handle_reservation(self):
        equipment = self.selected_equipment.get().strip()
        date_text = self.date_entry.get().strip()
        slot = self.selected_slot.get().strip()

        if not equipment or not date_text or not slot:
            messagebox.showerror("Reservation Error", "Please complete all booking fields.")
            return

        try:
            parsed_date = datetime.strptime(date_text, "%Y-%m-%d")
            formatted_date = parsed_date.strftime("%A, %d %B %Y")
        except ValueError:
            messagebox.showerror("Reservation Error", "Date must be in YYYY-MM-DD format.")
            return

        booking_number = len(self.bookings) + 1
        booking = {
            "equipment": equipment,
            "date": formatted_date,
            "slot": slot,
            "status": "Reserved",
            "checkin_code": f"GS-{booking_number:03d}-{parsed_date.strftime('%d%m')}",
        }

        self.bookings.append(booking)
        messagebox.showinfo("Reservation Confirmed", f"{equipment} reserved for {slot}.")
        self.show_member_dashboard("qr")

    def simulate_checkin(self):
        if not self.bookings:
            return

        latest_booking = self.bookings[-1]
        latest_booking["status"] = "Checked In"
        latest_booking["checked_in_at"] = datetime.now().strftime("%H:%M")
        messagebox.showinfo("Check-In Complete", "The member has been checked in successfully.")
        self.show_member_dashboard("qr")

    def show_staff_dashboard(self):
        page = self.create_scrollable_page()

        self.create_logged_in_navbar(
            page,
            f"Staff Dashboard — {self.current_staff_name}",
            f"{self.current_staff_role} · Monitor demand, utilisation, and operational insights.",
        )

        metrics = tk.Frame(page, bg=self.colors["bg"])
        metrics.pack(fill="x", padx=28, pady=(0, 18))
        self.create_metric_card(metrics, "Monthly Utilisation", "78%", "+6%")
        self.create_metric_card(metrics, "Peak Demand", "18:00–20:00", "Weekdays")
        self.create_metric_card(metrics, "No-Show Rate", "8.2%", "-2.1%")
        self.create_metric_card(metrics, "Recommended Upgrade", "1 New Rack", "High demand")

        body = tk.Frame(page, bg=self.colors["bg"])
        body.pack(fill="both", expand=True, padx=28, pady=(0, 28))

        left = tk.Frame(body, bg=self.colors["bg"])
        left.pack(side="left", fill="both", expand=True, padx=(0, 8))

        right = tk.Frame(body, bg=self.colors["bg"])
        right.pack(side="left", fill="both", expand=True, padx=(8, 0))

        self.build_staff_left(left)
        self.build_staff_right(right)

        self.bind_mousewheel_recursive(page)
        self.scroll_to_top()

    def build_staff_left(self, parent):
        util_panel = self.create_card(parent, bg=self.colors["panel"], padx=22, pady=22)
        util_panel.pack(fill="x", pady=(0, 16))

        self.create_section_title(util_panel, "Equipment Utilisation", "Estimated usage by equipment category.")
        tk.Frame(util_panel, bg=self.colors["panel"], height=16).pack()

        utilisation_data = [
            ("Squat Racks", 92),
            ("Benches", 84),
            ("Cable Machines", 71),
            ("Treadmills", 66),
            ("Spin Bikes", 58),
        ]
        for label, value in utilisation_data:
            self.create_bar_row(util_panel, label, value)

        heatmap_panel = self.create_card(parent, bg=self.colors["panel"], padx=22, pady=22)
        heatmap_panel.pack(fill="both", expand=True)

        self.create_section_title(heatmap_panel, "Peak Demand Heatmap", "Mock demand intensity across the week.")
        tk.Frame(heatmap_panel, bg=self.colors["panel"], height=16).pack()

        days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        times = ["06:00", "12:00", "17:00", "18:00", "19:00", "20:00", "21:00"]
        values = {
            "Mon": [28, 35, 72, 95, 100, 92, 58],
            "Tue": [26, 31, 70, 93, 98, 90, 54],
            "Wed": [24, 30, 69, 94, 97, 88, 53],
            "Thu": [25, 29, 71, 96, 99, 91, 55],
            "Fri": [22, 28, 64, 82, 86, 77, 49],
            "Sat": [40, 56, 61, 66, 63, 58, 45],
            "Sun": [34, 47, 53, 57, 54, 49, 38],
        }

        grid = tk.Frame(heatmap_panel, bg=self.colors["panel"])
        grid.pack(fill="both", expand=True)

        tk.Label(grid, text="", bg=self.colors["panel"], width=10).grid(row=0, column=0, padx=4, pady=4)

        for i, time_label in enumerate(times, start=1):
            tk.Label(
                grid,
                text=time_label,
                font=self.fonts["small"],
                fg=self.colors["text"],
                bg=self.colors["panel"],
                width=8,
            ).grid(row=0, column=i, padx=4, pady=4)

        for r, day in enumerate(days, start=1):
            tk.Label(
                grid,
                text=day,
                font=self.fonts["small"],
                fg=self.colors["text"],
                bg=self.colors["panel"],
                width=10,
            ).grid(row=r, column=0, padx=4, pady=4)

            for c, value in enumerate(values[day], start=1):
                tk.Label(
                    grid,
                    text=str(value),
                    font=("Helvetica", 9, "bold"),
                    fg="white",
                    bg=self.get_heatmap_color(value),
                    width=8,
                    height=2,
                ).grid(row=r, column=c, padx=4, pady=4)

    def build_staff_right(self, parent):
        insights = self.create_card(parent, bg=self.colors["panel"], padx=22, pady=22)
        insights.pack(fill="x", pady=(0, 16))

        self.create_section_title(insights, "Operator Insights", "Pitch-ready summary points for gym management.")
        tk.Frame(insights, bg=self.colors["panel"], height=16).pack()

        items = [
            ("High-Pressure Zone", "Squat racks exceed 90% usage during weekday evenings. Adding one extra rack is justified."),
            ("Underused Opportunity", "Cardio demand is steadier and lower during peak strength windows, suggesting layout optimisation potential."),
            ("Operational Benefit", "Equipment booking reduces uncertainty, improves member experience, and creates cleaner demand data."),
        ]

        for title, text in items:
            card = self.create_card(insights, bg=self.colors["panel_2"], padx=16, pady=16)
            card.pack(fill="x", pady=(0, 10))
            self.create_label(card, title, "card_title").pack(anchor="w")
            self.create_label(card, text, "body", self.colors["muted"], wraplength=420, justify="left", anchor="w").pack(anchor="w", pady=(8, 0))

        feed = self.create_card(parent, bg=self.colors["panel"], padx=22, pady=22)
        feed.pack(fill="both", expand=True)

        self.create_section_title(feed, "Recent Reservation Feed", "Live demo reservations created in this session.")
        tk.Frame(feed, bg=self.colors["panel"], height=16).pack()

        if not self.bookings:
            self.create_label(
                feed,
                "No live reservation activity yet in this demo session.",
                "body",
                self.colors["muted"],
                justify="left",
                anchor="w",
            ).pack(fill="x")
            return

        for booking in self.bookings:
            row = self.create_card(feed, bg=self.colors["panel_2"], padx=16, pady=16)
            row.pack(fill="x", pady=(0, 10))
            self.create_label(row, booking["equipment"], "card_title").pack(anchor="w")
            self.create_label(row, booking["date"], "body", self.colors["muted"]).pack(anchor="w", pady=(8, 0))
            self.create_label(row, booking["slot"], "body", self.colors["accent_blue"]).pack(anchor="w", pady=(4, 0))
            self.create_label(row, f"Status: {booking['status']}", "small", self.get_status_color_from_booking(booking)).pack(anchor="w", pady=(8, 0))

    def create_bar_row(self, parent, label, percentage):
        row = tk.Frame(parent, bg=self.colors["panel"])
        row.pack(fill="x", pady=6)

        self.create_label(row, label, "body", self.colors["text"], width=16, anchor="w").pack(side="left")

        bar_outer = tk.Frame(
            row,
            bg=self.colors["input_bg"],
            highlightthickness=1,
            highlightbackground=self.colors["border"],
            height=24,
        )
        bar_outer.pack(side="left", fill="x", expand=True, padx=14)

        width_units = max(1, int(percentage * 7))
        bar_inner = tk.Frame(bar_outer, bg=self.get_bar_color(percentage), width=width_units, height=22)
        bar_inner.pack(side="left", fill="y")

        self.create_label(row, f"{percentage}%", "body", self.colors["text"], width=6).pack(side="right")

    def get_bar_color(self, value):
        if value >= 85:
            return "#ef4444"
        if value >= 70:
            return "#f59e0b"
        return "#22c55e"

    def get_heatmap_color(self, value):
        if value >= 90:
            return "#c81e1e"
        if value >= 75:
            return "#ea580c"
        if value >= 60:
            return "#d97706"
        if value >= 45:
            return "#16a34a"
        return "#15803d"


def main():
    root = tk.Tk()
    GymSlotApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
