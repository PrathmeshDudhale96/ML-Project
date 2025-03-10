import streamlit as st
import pickle
import imaplib
import email
from email.header import decode_header

# Import the message submodule from the email package
from email import message_from_bytes

# Load the saved model, vectorizer, and label encoder
with open('spam_detection_model.pkl', 'rb') as model_file:
    loaded_model = pickle.load(model_file)

with open('tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
    loaded_vectorizer = pickle.load(vectorizer_file)

with open('label_encoder.pkl', 'rb') as encoder_file:
    loaded_encoder = pickle.load(encoder_file)

def get_email_body(msg):
    """Extract the body of the email."""
    if msg.is_multipart():
        for part in msg.walk():
            content_type = part.get_content_type()
            content_disposition = str(part.get("Content-Disposition"))
            if content_type == "text/plain" and "attachment" not in content_disposition:
                return part.get_payload(decode=True).decode()  # Decode the body
    else:
        content_type = msg.get_content_type()
        if content_type == "text/plain":
            return msg.get_payload(decode=True).decode()  # Decode the body
    return None  # Return None if no plain text body is found

def fetch_latest_emails(username, password, imap_server='imap.gmail.com', num_emails=10):
    # Connect to the server
    mail = imaplib.IMAP4_SSL(imap_server)

    # Login to the email account
    mail.login(username, password)

    # Select the mailbox you want to check
    mail.select("inbox")

    # Search for all emails
    status, messages = mail.search(None, 'ALL')

    # Get email IDs
    email_ids = messages[0].split()

    # Select the latest `num_emails` emails (limit the list to the last `num_emails` entries)
    latest_email_ids = email_ids[-num_emails:]

    # List to hold the emails
    emails = []

    # Fetch the latest emails
    for email_id in latest_email_ids:
        _, msg_data = mail.fetch(email_id, '(RFC822)')
        for response_part in msg_data:
            if isinstance(response_part, tuple):
                # Use message_from_bytes directly
                msg = message_from_bytes(response_part[1])
                subject, encoding = decode_header(msg["Subject"])[0]
                if isinstance(subject, bytes):
                    subject = subject.decode(encoding if encoding else 'utf-8')
                from_ = msg.get("From")
                body = get_email_body(msg)  # Assuming get_email_body is defined elsewhere
                # Check if body is None before appending to emails
                if body is not None:
                    emails.append({'subject': subject, 'from': from_, 'body': body})

    return emails

def classify_emails(emails, model, vectorizer):
    """Classify emails as spam or ham"""
    # Filter out emails with None body
    email_bodies = [email['body'] for email in emails if email['body'] is not None]
    email_tfidf = vectorizer.transform(email_bodies)
    predictions = model.predict(email_tfidf)
    return predictions

# Streamlit app
def main():
    st.title("Spam Email Classifier")

    # Get user input
    username = st.text_input("Enter your email address")
    password = st.text_input("Enter your password", type="password")
    num_emails = st.number_input("Number of emails to fetch", min_value=1, max_value=100, value=10)

    if st.button("Classify Emails"):
        try:
            # Fetch emails
            emails = fetch_latest_emails(username, password, num_emails=num_emails)

            if emails:
                # Classify emails
                predictions = classify_emails(emails, loaded_model, loaded_vectorizer)

                # Display results
                for email, prediction in zip(emails, predictions):
                    label = 'a Spam' if prediction == 1 else 'Real'

                    if prediction == 1:
                        # Spam -> red text with yellow background
                        bold_highlighted_label = label   
                    else:
                        # Real -> green text with yellow background
                        bold_highlighted_label =  label  
                    st.write(
                        f"Email from **{email['from']}** with subject '**{email['subject']}'** is classified as "
                        f"<span style='color: red; font-weight: bold;'>{bold_highlighted_label}</span>.",
                        unsafe_allow_html=True
                    )
            else:
                st.warning("No emails found.")

        except imaplib.IMAP4.error as e:
            st.error(f"Failed to fetch emails: {e}")

if __name__ == "__main__":
    main()