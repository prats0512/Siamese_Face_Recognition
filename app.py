import tkinter as tk
from tkinter import messagebox, simpledialog
import os
import json

from onboarding import enroll_user
from face_recognition_system import start_recognition


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

REGISTERED_USERS_PATH = os.path.join(
    BASE_DIR,
    "registered_users"
)


def view_registered_users():

    users = []

    if os.path.exists(REGISTERED_USERS_PATH):

        for filename in os.listdir(REGISTERED_USERS_PATH):

            if filename.endswith(".json"):

                filepath = os.path.join(
                    REGISTERED_USERS_PATH,
                    filename
                )

                try:

                    with open(filepath, "r") as file:
                        data = json.load(file)

                    users.append(
                        f"ID: {data['user_id']}    "
                        f"Name: {data['name']}"
                    )

                except Exception:
                    pass

    if not users:

        messagebox.showinfo(
            "Registered Users",
            "No registered users found."
        )

        return

    messagebox.showinfo(
        "Registered Users",
        "\n".join(users)
    )


def register_new_user():

    user_id = simpledialog.askstring(
        "Register User",
        "Enter User ID:"
    )

    if not user_id:
        return

    name = simpledialog.askstring(
        "Register User",
        "Enter Name:"
    )

    if not name:
        return

    user_file = os.path.join(
        REGISTERED_USERS_PATH,
        f"{user_id}.json"
    )

    if os.path.exists(user_file):

        messagebox.showerror(
            "Registration Error",
            f"User ID '{user_id}' already exists."
        )

        return

    root.withdraw()

    try:

        success = enroll_user(
            user_id=user_id,
            name=name,
            required_samples=5
        )

        if success:

            messagebox.showinfo(
                "Registration Successful",
                f"{name} has been registered successfully."
            )

        else:

            messagebox.showwarning(
                "Registration Failed",
                "User registration was not completed."
            )

    finally:

        root.deiconify()


def start_face_recognition():

    root.withdraw()

    try:

        start_recognition()

    finally:

        root.deiconify()


def delete_registered_user():

    user_id = simpledialog.askstring(
        "Delete User",
        "Enter User ID to delete:"
    )

    if not user_id:
        return

    user_file = os.path.join(
        REGISTERED_USERS_PATH,
        f"{user_id}.json"
    )

    if not os.path.exists(user_file):

        messagebox.showerror(
            "Delete User",
            f"No registered user found with ID '{user_id}'."
        )

        return

    confirmation = messagebox.askyesno(
        "Confirm Deletion",
        f"Delete registered user '{user_id}'?"
    )

    if confirmation:

        os.remove(user_file)

        messagebox.showinfo(
            "Delete User",
            f"User '{user_id}' deleted successfully."
        )


def exit_application():

    root.destroy()


root = tk.Tk()

root.title(
    "Siamese Neural Network Face Recognition"
)

root.geometry("600x500")

root.resizable(
    False,
    False
)


title_label = tk.Label(
    root,
    text="Siamese Neural Network\nFace Recognition System",
    font=("Arial", 22, "bold"),
    pady=25
)

title_label.pack()


subtitle_label = tk.Label(
    root,
    text="Face Registration and Real-Time Verification",
    font=("Arial", 11)
)

subtitle_label.pack(
    pady=(0, 25)
)


button_frame = tk.Frame(root)

button_frame.pack()


button_options = {
    "font": ("Arial", 13),
    "width": 28,
    "height": 2,
    "pady": 5
}


tk.Button(
    button_frame,
    text="Register New User",
    command=register_new_user,
    **button_options
).pack()


tk.Button(
    button_frame,
    text="Start Face Recognition",
    command=start_face_recognition,
    **button_options
).pack()


tk.Button(
    button_frame,
    text="View Registered Users",
    command=view_registered_users,
    **button_options
).pack()


tk.Button(
    button_frame,
    text="Delete Registered User",
    command=delete_registered_user,
    **button_options
).pack()


tk.Button(
    button_frame,
    text="Exit",
    command=exit_application,
    **button_options
).pack()


footer = tk.Label(
    root,
    text="Powered by TensorFlow, OpenCV and Siamese Neural Network",
    font=("Arial", 9),
    pady=20
)

footer.pack(
    side="bottom"
)


root.mainloop()
