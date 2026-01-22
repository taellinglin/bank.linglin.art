"""
Email service for sending verification and notification emails
"""
from flask_mail import Mail, Message
from flask import render_template_string, url_for
import os
from datetime import datetime

mail = Mail()

APP_NAME = "Ling Country Treasury"
APP_SHORT = "Linglin Treasury"

def _get_app_url() -> str:
    return os.environ.get("APP_URL", "http://localhost:5000").rstrip("/")

def _build_email_html(
    title: str,
    preheader: str,
    heading: str,
    subtitle: str,
    content_html: str,
    cta_text: str = None,
    cta_url: str = None,
):
    button_html = ""
    if cta_text and cta_url:
        button_html = f"""
            <tr>
                <td align=\"center\" style=\"padding: 24px 0 0;\">
                    <a href=\"{cta_url}\" style=\"display: inline-block; padding: 14px 28px; background: linear-gradient(135deg, #3B82F6 0%, #22D3EE 100%); color: #0b0f14; text-decoration: none; font-weight: 700; border-radius: 999px; letter-spacing: 0.2px;\">{cta_text}</a>
                </td>
            </tr>
        """

    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset=\"UTF-8\">
        <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">
        <title>{title}</title>
    </head>
    <body style=\"margin:0; padding:0; background-color:#0b0f14; color:#e6edf3; font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Arial, sans-serif;\">
        <span style=\"display:none; visibility:hidden; opacity:0; height:0; width:0;\">{preheader}</span>
        <table role=\"presentation\" width=\"100%\" cellpadding=\"0\" cellspacing=\"0\" style=\"background-color:#0b0f14; padding:32px 16px;\">
            <tr>
                <td align=\"center\">
                    <table role=\"presentation\" width=\"600\" cellpadding=\"0\" cellspacing=\"0\" style=\"max-width:600px; width:100%; background:#0f172a; border:1px solid #1f2937; border-radius:20px; overflow:hidden; box-shadow:0 16px 40px rgba(0,0,0,0.45);\">
                        <tr>
                            <td style=\"padding:28px 32px; background:linear-gradient(135deg, #111827 0%, #0b1220 100%); border-bottom:1px solid #1f2937;\">
                                <div style=\"font-size:14px; letter-spacing:2px; text-transform:uppercase; color:#94a3b8; font-weight:600;\">{APP_NAME}</div>
                                <h1 style=\"margin:10px 0 6px; font-size:26px; color:#f8fafc;\">{heading}</h1>
                                <p style=\"margin:0; color:#cbd5f5; font-size:15px;\">{subtitle}</p>
                            </td>
                        </tr>
                        <tr>
                            <td style=\"padding:30px 32px; background:#0f172a;\">
                                {content_html}
                                {button_html}
                            </td>
                        </tr>
                        <tr>
                            <td style=\"padding:20px 32px 28px; color:#64748b; font-size:12px; border-top:1px dashed #1f2937;\">
                                This email was sent by {APP_NAME}. If you did not request this, you can safely ignore it.
                            </td>
                        </tr>
                    </table>
                </td>
            </tr>
        </table>
    </body>
    </html>
    """

def init_mail(app):
    """Initialize Flask-Mail with app configuration"""
    # Load from environment variables
    app.config['MAIL_SERVER'] = os.environ.get('MAIL_SERVER', 'smtp.gmail.com')
    app.config['MAIL_PORT'] = int(os.environ.get('MAIL_PORT', 587))
    app.config['MAIL_USE_TLS'] = os.environ.get('MAIL_USE_TLS', 'true').lower() == 'true'
    app.config['MAIL_USE_SSL'] = os.environ.get('MAIL_USE_SSL', 'false').lower() == 'true'
    app.config['MAIL_USERNAME'] = os.environ.get('MAIL_USERNAME')
    app.config['MAIL_PASSWORD'] = os.environ.get('MAIL_PASSWORD')
    app.config['MAIL_DEFAULT_SENDER'] = os.environ.get('MAIL_DEFAULT_SENDER')
    
    mail.init_app(app)
    return mail

def send_banknote_generation_started_notification(user_email, username, task_id=None):
    """Send email notification when banknote generation starts"""
    subject = "Ling Country Treasury — Generation started"
    app_url = _get_app_url()
    dashboard_url = f"{app_url}/profile/{username}" if username else f"{app_url}/dashboard"

    task_line = f"<li><strong>Task ID:</strong> #{task_id}</li>" if task_id else ""
    content_html = f"""
        <p style=\"margin:0 0 16px; color:#e2e8f0;\">Hi {username},</p>
        <div style=\"background:#0b1220; border:1px solid #1f2937; padding:16px; border-radius:14px;\">
            <p style=\"margin:0; color:#e2e8f0;\"><strong>Generation is now in progress.</strong></p>
            <ul style=\"margin:12px 0 0; padding-left:18px; color:#cbd5f5;\">
                {task_line}
                <li>Denominations: 1 → 100,000,000 LKC</li>
                <li>Status: Processing</li>
            </ul>
        </div>
        <p style=\"margin:16px 0 0; color:#94a3b8;\">You can follow progress on your profile page.</p>
    """

    html_body = _build_email_html(
        title=subject,
        preheader="Your banknote generation has started.",
        heading="Generation started",
        subtitle="We are minting your new banknotes",
        content_html=content_html,
        cta_text="View progress",
        cta_url=dashboard_url,
    )
    
    text_body = f"""
    {APP_NAME} — Generation started

    Hi {username},

    Your banknote generation has started.
    Task ID: #{task_id if task_id else ''}
    {dashboard_url}
    """

    try:
        msg = Message(
            subject=subject,
            recipients=[user_email],
        )
        msg.body = text_body
        msg.html = html_body
        mail.send(msg)
        print(f"[EMAIL] Sent generation-started notification to {user_email}")
        return True
    except Exception as e:
        print(f"[EMAIL ERROR] Failed to send generation-started notification: {e}")
        return False

def send_email_change_verification(new_email, username, verification_token, old_email, app_url='http://localhost:5000'):
    """Send email verification when user changes their email address"""
    verification_url = f"{app_url}/profile/verify-email-change/{verification_token}"
    subject = "Ling Country Treasury — Confirm your new email"

    content_html = f"""
        <p style=\"margin:0 0 16px; color:#e2e8f0;\">Hi {username},</p>
        <div style=\"background:#0b1220; border:1px solid #1f2937; padding:16px; border-radius:14px;\">
            <p style=\"margin:0; color:#e2e8f0;\"><strong>Email change requested</strong></p>
            <p style=\"margin:10px 0 0; color:#cbd5f5;\">Old email: <span style=\"color:#f8fafc;\">{old_email}</span></p>
            <p style=\"margin:6px 0 0; color:#cbd5f5;\">New email: <span style=\"color:#f8fafc;\">{new_email}</span></p>
        </div>
        <p style=\"margin:16px 0 0; color:#94a3b8;\">Confirm this change to keep your account secure.</p>
        <p style=\"margin:12px 0 0; color:#94a3b8;\">Link expires in 24 hours.</p>
        <p style=\"margin:12px 0 0; color:#94a3b8; word-break: break-all;\">{verification_url}</p>
    """

    html_body = _build_email_html(
        title=subject,
        preheader="Confirm your new email address.",
        heading="Confirm your new email",
        subtitle="Secure your account change",
        content_html=content_html,
        cta_text="Confirm email",
        cta_url=verification_url,
    )
    
    try:
        msg = Message(
            subject=subject,
            recipients=[new_email],
            html=html_body
        )
        mail.send(msg)
        print(f"[EMAIL] Sent email change verification to {new_email}")
        return True
    except Exception as e:
        print(f"[EMAIL ERROR] Failed to send email change verification: {e}")
        return False

def send_verification_email(user_email, username, verification_token, app_url='http://localhost:5000'):
    """Send email verification email to new user"""
    verification_url = f"{app_url}/verify-email/{verification_token}"
    subject = "Ling Country Treasury — Verify your email"

    content_html = f"""
        <p style=\"margin:0 0 16px; color:#e2e8f0;\">Hi {username},</p>
        <p style=\"margin:0 0 16px; color:#cbd5f5;\">Thanks for joining {APP_NAME}. Please verify your email to activate your account.</p>
        <div style=\"background:#0b1220; border:1px solid #1f2937; padding:16px; border-radius:14px;\">
            <p style=\"margin:0; color:#e2e8f0;\"><strong>Verification link</strong></p>
            <p style=\"margin:10px 0 0; color:#94a3b8; word-break: break-all;\">{verification_url}</p>
            <p style=\"margin:10px 0 0; color:#94a3b8;\">Link expires in 24 hours.</p>
        </div>
    """

    html_body = _build_email_html(
        title=subject,
        preheader="Verify your email to activate your account.",
        heading="Verify your email",
        subtitle="Activate your Ling Country Treasury account",
        content_html=content_html,
        cta_text="Verify email",
        cta_url=verification_url,
    )

    text_body = f"""
    {APP_NAME} — Verify your email

    Hi {username},

    Please verify your email to activate your account:
    {verification_url}

    This link expires in 24 hours.
    """
    
    try:
        msg = Message(subject, recipients=[user_email])
        msg.body = text_body
        msg.html = html_body
        mail.send(msg)
        return True
    except Exception as e:
        print(f"[ERROR] Failed to send verification email: {e}")
        return False

def send_banknote_generation_notification(user_email, username, denomination, count=1, serial_numbers=None):
    """Send email notification when banknotes are generated"""
    subject = f"Ling Country Treasury — {count} banknote(s) ready"
    app_url = _get_app_url()
    dashboard_url = f"{app_url}/profile/{username}" if username else f"{app_url}/dashboard"
    
    serial_list_html = ""
    if serial_numbers:
        serial_list_html = "<ul>"
        for sn in serial_numbers[:5]:  # Show first 5
            serial_list_html += f"<li><code>{sn}</code></li>"
        if len(serial_numbers) > 5:
            serial_list_html += f"<li>... and {len(serial_numbers) - 5} more</li>"
        serial_list_html += "</ul>"
    
    serials_block = (
        f"<div style=\"margin-top:16px; color:#94a3b8;\"><strong>Serials:</strong>{serial_list_html}</div>"
        if serial_numbers
        else ""
    )

    content_html = f"""
        <p style=\"margin:0 0 16px; color:#e2e8f0;\">Hi {username},</p>
        <div style=\"background:#0b1220; border:1px solid #1f2937; padding:16px; border-radius:14px;\">
            <p style=\"margin:0; color:#e2e8f0;\"><strong>Generation complete</strong></p>
            <ul style=\"margin:12px 0 0; padding-left:18px; color:#cbd5f5;\">
                <li>Denomination: {denomination} LKC</li>
                <li>Count: {count}</li>
                <li>Time: {datetime.now().strftime('%Y-%m-%d %H:%M')}</li>
            </ul>
        </div>
        {serials_block}
        <p style=\"margin:16px 0 0; color:#94a3b8;\">You can download your banknotes from your profile.</p>
    """

    html_body = _build_email_html(
        title=subject,
        preheader="Your banknotes are ready.",
        heading="Banknotes ready",
        subtitle="Your generation finished successfully",
        content_html=content_html,
        cta_text="View banknotes",
        cta_url=dashboard_url,
    )
    
    text_body = f"""
    {APP_NAME} — Banknotes ready

    Hi {username},

    Your banknotes are ready.
    Denomination: {denomination} LKC
    Count: {count}
    Time: {datetime.now().strftime('%Y-%m-%d %H:%M')}

    View them here:
    {dashboard_url}
    """
    
    try:
        msg = Message(subject, recipients=[user_email])
        msg.body = text_body
        msg.html = html_body
        mail.send(msg)
        return True
    except Exception as e:
        print(f"[ERROR] Failed to send banknote notification email: {e}")
        return False

def send_generation_ready_notification(user_email, username, days_until_next=0):
    """Send email notification when user can generate banknotes again"""
    subject = "Ling Country Treasury — Generation is available"
    app_url = _get_app_url()
    dashboard_url = f"{app_url}/profile/{username}" if username else f"{app_url}/dashboard"

    content_html = f"""
        <p style=\"margin:0 0 16px; color:#e2e8f0;\">Hi {username},</p>
        <div style=\"background:#0b1220; border:1px solid #1f2937; padding:16px; border-radius:14px;\">
            <p style=\"margin:0; color:#e2e8f0;\"><strong>You can generate new banknotes again.</strong></p>
            <p style=\"margin:10px 0 0; color:#94a3b8;\">Your cooldown has ended.</p>
        </div>
        <p style=\"margin:16px 0 0; color:#94a3b8;\">Open your profile to start a new generation.</p>
    """

    html_body = _build_email_html(
        title=subject,
        preheader="You can generate new banknotes again.",
        heading="Generation available",
        subtitle="Your cooldown has ended",
        content_html=content_html,
        cta_text="Generate now",
        cta_url=dashboard_url,
    )

    text_body = f"""
    {APP_NAME} — Generation available

    Hi {username},

    Your cooldown has ended. You can generate new banknotes again.
    {dashboard_url}
    """
    
    try:
        msg = Message(subject, recipients=[user_email])
        msg.body = text_body
        msg.html = html_body
        mail.send(msg)
        return True
    except Exception as e:
        print(f"[ERROR] Failed to send generation ready notification: {e}")
        return False

def send_generation_completed_notification(user_email, username, status, total_pairs=0, task_id=None):
    """Send email notification when generation finishes (completed/partial)."""
    subject = f"Ling Country Treasury — Generation {status}"
    app_url = _get_app_url()
    dashboard_url = f"{app_url}/profile/{username}" if username else f"{app_url}/dashboard"

    status_label = "Completed" if status == "completed" else "Partial"
    task_line = f"<li><strong>Task ID:</strong> #{task_id}</li>" if task_id else ""

    content_html = f"""
        <p style=\"margin:0 0 16px; color:#e2e8f0;\">Hi {username},</p>
        <div style=\"background:#0b1220; border:1px solid #1f2937; padding:16px; border-radius:14px;\">
            <p style=\"margin:0; color:#e2e8f0;\"><strong>Generation {status_label}</strong></p>
            <ul style=\"margin:12px 0 0; padding-left:18px; color:#cbd5f5;\">
                {task_line}
                <li>Total pairs created: {total_pairs}</li>
                <li>Status: {status_label}</li>
            </ul>
        </div>
        <p style=\"margin:16px 0 0; color:#94a3b8;\">Open your profile to review the results.</p>
    """

    html_body = _build_email_html(
        title=subject,
        preheader=f"Generation {status_label.lower()}.",
        heading=f"Generation {status_label}",
        subtitle="Your banknote generation has finished",
        content_html=content_html,
        cta_text="View results",
        cta_url=dashboard_url,
    )

    text_body = f"""
    {APP_NAME} — Generation {status_label}

    Hi {username},

    Your generation finished with status: {status_label}.
    Total pairs created: {total_pairs}
    {dashboard_url}
    """

    try:
        msg = Message(subject, recipients=[user_email])
        msg.body = text_body
        msg.html = html_body
        mail.send(msg)
        return True
    except Exception as e:
        print(f"[ERROR] Failed to send generation completed notification: {e}")
        return False

def send_generation_failed_notification(user_email, username, error_message=None, task_id=None):
    """Send email notification when generation fails."""
    subject = "Ling Country Treasury — Generation failed"
    app_url = _get_app_url()
    dashboard_url = f"{app_url}/profile/{username}" if username else f"{app_url}/dashboard"

    task_line = f"<li><strong>Task ID:</strong> #{task_id}</li>" if task_id else ""
    error_line = f"<li><strong>Error:</strong> {error_message}</li>" if error_message else ""

    content_html = f"""
        <p style=\"margin:0 0 16px; color:#e2e8f0;\">Hi {username},</p>
        <div style=\"background:#2b0f12; border:1px solid #7f1d1d; padding:16px; border-radius:14px;\">
            <p style=\"margin:0; color:#fee2e2;\"><strong>Generation failed</strong></p>
            <ul style=\"margin:12px 0 0; padding-left:18px; color:#fecaca;\">
                {task_line}
                {error_line}
            </ul>
        </div>
        <p style=\"margin:16px 0 0; color:#94a3b8;\">You can retry from your profile.</p>
    """

    html_body = _build_email_html(
        title=subject,
        preheader="Generation failed. Review details.",
        heading="Generation failed",
        subtitle="We ran into an error",
        content_html=content_html,
        cta_text="Open profile",
        cta_url=dashboard_url,
    )

    text_body = f"""
    {APP_NAME} — Generation failed

    Hi {username},

    Your generation failed.
    {error_message or ''}
    {dashboard_url}
    """

    try:
        msg = Message(subject, recipients=[user_email])
        msg.body = text_body
        msg.html = html_body
        mail.send(msg)
        return True
    except Exception as e:
        print(f"[ERROR] Failed to send generation failed notification: {e}")
        return False
