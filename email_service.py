"""
Email service for sending verification and notification emails
"""
from flask_mail import Mail, Message
from flask import render_template_string, url_for
import os
from datetime import datetime

mail = Mail()

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
    subject = "灵国国库 - バンクノート生成開始 | Generation Started"
    
    html_body = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <style>
            body {{
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
                line-height: 1.6;
                color: #333;
                max-width: 600px;
                margin: 0 auto;
                padding: 20px;
            }}
            .container {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                border-radius: 10px;
                padding: 30px;
                color: white;
            }}
            .content {{
                background: white;
                padding: 30px;
                border-radius: 5px;
                margin-top: 20px;
            }}
            .button {{
                display: inline-block;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 12px 30px;
                text-decoration: none;
                border-radius: 5px;
                margin: 20px 0;
            }}
            .info-box {{
                background: #e3f2fd;
                border-left: 4px solid #2196f3;
                padding: 15px;
                margin: 20px 0;
            }}
            .footer {{
                margin-top: 20px;
                font-size: 12px;
                color: #666;
                text-align: center;
            }}
            .logo {{
                font-size: 24px;
                font-weight: bold;
                margin-bottom: 10px;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="logo">💵 灵国国库</div>
            <h2>バンクノート生成を開始しました</h2>
        </div>
        
        <div class="content">
            <h3>{username}さん、</h3>
            
            <div class="info-box">
                <h4>🚀 生成開始</h4>
                <p>バンクノートの生成プロセスを開始しました!</p>
                {f"<p><strong>タスクID:</strong> #{task_id}</p>" if task_id else ""}
            </div>
            
            <p>全ての額面(1, 10, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000 卢纳币)のバンクノートを生成中です。</p>
            
            <p>生成には数分かかる場合があります。完了次第、別途メールでお知らせします。</p>
            
            <a href="http://localhost:5000/dashboard" class="button">進捗を確認</a>
            
            <p><small>💡 ヒント: ダッシュボードで生成状況をリアルタイムで確認できます。</small></p>
        </div>
        
        <div class="footer">
            <p>灵国国库 - Ling Country Treasury</p>
            <p>このメールは自動送信されています</p>
        </div>
    </body>
    </html>
    """
    
    try:
        msg = Message(
            subject=subject,
            recipients=[user_email],
            html=html_body
        )
        mail.send(msg)
        print(f"[EMAIL] Sent generation-started notification to {user_email}")
        return True
    except Exception as e:
        print(f"[EMAIL ERROR] Failed to send generation-started notification: {e}")
        return False

def send_email_change_verification(new_email, username, verification_token, old_email, app_url='http://localhost:5000'):
    """Send email verification when user changes their email address"""
    verification_url = f"{app_url}/profile/verify-email-change/{verification_token}"
    
    subject = "灵国国库 - メールアドレス変更の確認 | Verify Email Change"
    
    html_body = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <style>
            body {{
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
                line-height: 1.6;
                color: #333;
                max-width: 600px;
                margin: 0 auto;
                padding: 20px;
            }}
            .container {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                border-radius: 10px;
                padding: 30px;
                color: white;
            }}
            .content {{
                background: white;
                border-radius: 8px;
                padding: 30px;
                margin-top: 20px;
                color: #333;
            }}
            .button {{
                display: inline-block;
                padding: 15px 40px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                text-decoration: none;
                border-radius: 5px;
                margin: 20px 0;
                font-weight: bold;
            }}
            .warning-box {{
                background: #fff3cd;
                border: 1px solid #ffc107;
                border-radius: 5px;
                padding: 15px;
                margin: 20px 0;
            }}
            .info-box {{
                background: #e3f2fd;
                border-left: 4px solid #2196f3;
                padding: 15px;
                margin: 20px 0;
            }}
            code {{
                background: #f4f4f4;
                padding: 2px 6px;
                border-radius: 3px;
                font-family: monospace;
            }}
            .footer {{
                margin-top: 20px;
                font-size: 12px;
                color: #666;
                text-align: center;
            }}
            .logo {{
                font-size: 24px;
                font-weight: bold;
                margin-bottom: 10px;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="logo">💵 灵国国库</div>
            <h2>メールアドレス変更の確認</h2>
        </div>
        
        <div class="content">
            <h3>{username}さん、</h3>
            
            <div class="warning-box">
                <h4>⚠️ メールアドレス変更リクエスト</h4>
                <p>あなたのアカウントでメールアドレスの変更が要求されました。</p>
            </div>
            
            <div class="info-box">
                <p><strong>旧メールアドレス:</strong> <code>{old_email}</code></p>
                <p><strong>新メールアドレス:</strong> <code>{new_email}</code></p>
            </div>
            
            <p>この変更を完了するには、下のボタンをクリックして新しいメールアドレスを確認してください:</p>
            
            <a href="{verification_url}" class="button">メールアドレスを確認</a>
            
            <p><small>リンクをクリックできない場合は、以下のURLをブラウザにコピー&ペーストしてください:</small></p>
            <p><code style="word-break: break-all;">{verification_url}</code></p>
            
            <div class="warning-box">
                <p><strong>⏰ 注意:</strong> このリンクは24時間有効です。</p>
                <p><strong>🔒 セキュリティ:</strong> このリクエストに心当たりがない場合は、このメールを無視してください。アカウントのパスワードを変更することをお勧めします。</p>
            </div>
        </div>
        
        <div class="footer">
            <p>灵国国库 - Ling Country Treasury</p>
            <p>このメールは自動送信されています</p>
        </div>
    </body>
    </html>
    """
    
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
    
    subject = "灵国国库 - メールアドレスの確認 | Verify Your Email"
    
    html_body = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <style>
            body {{
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
                line-height: 1.6;
                color: #333;
                max-width: 600px;
                margin: 0 auto;
                padding: 20px;
            }}
            .container {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                border-radius: 10px;
                padding: 30px;
                color: white;
            }}
            .content {{
                background: white;
                border-radius: 8px;
                padding: 30px;
                margin-top: 20px;
                color: #333;
            }}
            .button {{
                display: inline-block;
                padding: 12px 30px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                text-decoration: none;
                border-radius: 5px;
                margin: 20px 0;
                font-weight: bold;
            }}
            .footer {{
                margin-top: 20px;
                font-size: 12px;
                color: #666;
                text-align: center;
            }}
            .logo {{
                font-size: 24px;
                font-weight: bold;
                margin-bottom: 10px;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="logo">🏦 灵国国库</div>
            <h2>Welcome to Ling Country Treasury!</h2>
        </div>
        
        <div class="content">
            <h3>こんにちは {username}さん、</h3>
            
            <p>灵国国库へようこそ！アカウント登録ありがとうございます。</p>
            
            <p>メールアドレスを確認するには、以下のボタンをクリックしてください：</p>
            
            <a href="{verification_url}" class="button">メールアドレスを確認</a>
            
            <p>または、以下のリンクをブラウザにコピー＆ペーストしてください：</p>
            <p style="word-break: break-all; color: #667eea;">{verification_url}</p>
            
            <hr style="margin: 30px 0; border: none; border-top: 1px solid #eee;">
            
            <h4>📋 次のステップ：</h4>
            <ul>
                <li>メールアドレスの確認</li>
                <li>プロフィール画像のアップロード</li>
                <li>バンクノートの生成開始</li>
            </ul>
            
            <p><strong>注意：</strong> このリンクは24時間有効です。</p>
        </div>
        
        <div class="footer">
            <p>このメールに心当たりがない場合は、無視してください。</p>
            <p>© 2026 灵国国库 | Ling Country Treasury</p>
        </div>
    </body>
    </html>
    """
    
    text_body = f"""
    灵国国库 - メールアドレスの確認
    
    こんにちは {username}さん、
    
    灵国国库へようこそ！
    
    メールアドレスを確認するには、以下のリンクをクリックしてください：
    {verification_url}
    
    このリンクは24時間有効です。
    
    このメールに心当たりがない場合は、無視してください。
    
    © 2026 灵国国库 | Ling Country Treasury
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
    subject = f"灵国国库 - バンクノート生成完了 | {count}枚のバンクノートが利用可能です"
    
    serial_list_html = ""
    if serial_numbers:
        serial_list_html = "<ul>"
        for sn in serial_numbers[:5]:  # Show first 5
            serial_list_html += f"<li><code>{sn}</code></li>"
        if len(serial_numbers) > 5:
            serial_list_html += f"<li>... and {len(serial_numbers) - 5} more</li>"
        serial_list_html += "</ul>"
    
    html_body = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <style>
            body {{
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
                line-height: 1.6;
                color: #333;
                max-width: 600px;
                margin: 0 auto;
                padding: 20px;
            }}
            .container {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                border-radius: 10px;
                padding: 30px;
                color: white;
            }}
            .content {{
                background: white;
                border-radius: 8px;
                padding: 30px;
                margin-top: 20px;
                color: #333;
            }}
            .button {{
                display: inline-block;
                padding: 12px 30px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                text-decoration: none;
                border-radius: 5px;
                margin: 20px 0;
                font-weight: bold;
            }}
            .success-box {{
                background: #d4edda;
                border: 1px solid #c3e6cb;
                border-radius: 5px;
                padding: 15px;
                margin: 20px 0;
            }}
            code {{
                background: #f4f4f4;
                padding: 2px 6px;
                border-radius: 3px;
                font-family: monospace;
            }}
            .footer {{
                margin-top: 20px;
                font-size: 12px;
                color: #666;
                text-align: center;
            }}
            .logo {{
                font-size: 24px;
                font-weight: bold;
                margin-bottom: 10px;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="logo">💵 灵国国库</div>
            <h2>バンクノート生成完了！</h2>
        </div>
        
        <div class="content">
            <h3>{username}さん、</h3>
            
            <div class="success-box">
                <h4>✅ 生成完了</h4>
                <p><strong>{count}枚</strong>のバンクノートが正常に生成されました！</p>
            </div>
            
            <h4>📋 生成詳細：</h4>
            <ul>
                <li><strong>額面：</strong> {denomination} 卢纳币</li>
                <li><strong>枚数：</strong> {count}枚</li>
                <li><strong>生成日時：</strong> {datetime.now().strftime('%Y年%m月%d日 %H:%M')}</li>
            </ul>
            
            {f"<h4>🔢 シリアル番号：</h4>{serial_list_html}" if serial_numbers else ""}
            
            <a href="http://localhost:5000/dashboard" class="button">ダッシュボードで確認</a>
            
            <p>バンクノートはダッシュボードからダウンロードできます。</p>
        </div>
        
        <div class="footer">
            <p>© 2026 灵国国库 | Ling Country Treasury</p>
        </div>
    </body>
    </html>
    """
    
    text_body = f"""
    灵国国库 - バンクノート生成完了
    
    {username}さん、
    
    {count}枚のバンクノートが正常に生成されました！
    
    生成詳細：
    - 額面：{denomination} 卢纳币
    - 枚数：{count}枚
    - 生成日時：{datetime.now().strftime('%Y年%m月%d日 %H:%M')}
    
    バンクノートはダッシュボードからダウンロードできます。
    http://localhost:5000/dashboard
    
    © 2026 灵国国库 | Ling Country Treasury
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
    subject = "灵国国库 - バンクノート生成が可能になりました！"
    
    html_body = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <style>
            body {{
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
                line-height: 1.6;
                color: #333;
                max-width: 600px;
                margin: 0 auto;
                padding: 20px;
            }}
            .container {{
                background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
                border-radius: 10px;
                padding: 30px;
                color: white;
            }}
            .content {{
                background: white;
                border-radius: 8px;
                padding: 30px;
                margin-top: 20px;
                color: #333;
            }}
            .button {{
                display: inline-block;
                padding: 12px 30px;
                background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
                color: white;
                text-decoration: none;
                border-radius: 5px;
                margin: 20px 0;
                font-weight: bold;
            }}
            .info-box {{
                background: #d1ecf1;
                border: 1px solid #bee5eb;
                border-radius: 5px;
                padding: 15px;
                margin: 20px 0;
            }}
            .footer {{
                margin-top: 20px;
                font-size: 12px;
                color: #666;
                text-align: center;
            }}
            .logo {{
                font-size: 24px;
                font-weight: bold;
                margin-bottom: 10px;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="logo">🎉 灵国国库</div>
            <h2>バンクノート生成が可能です！</h2>
        </div>
        
        <div class="content">
            <h3>{username}さん、</h3>
            
            <div class="info-box">
                <h4>ℹ️ お知らせ</h4>
                <p>クールダウン期間が終了し、新しいバンクノートを生成できるようになりました！</p>
            </div>
            
            <p>今すぐダッシュボードにアクセスして、新しいバンクノートを生成しましょう。</p>
            
            <a href="http://localhost:5000/dashboard" class="button">バンクノートを生成</a>
            
            <h4>💡 ヒント：</h4>
            <ul>
                <li>高品質なプロフィール画像を使用すると、より美しいバンクノートが生成されます</li>
                <li>複数の額面のバンクノートを一度に生成できます</li>
                <li>生成されたバンクノートはブロックチェーンに記録されます</li>
            </ul>
        </div>
        
        <div class="footer">
            <p>© 2026 灵国国库 | Ling Country Treasury</p>
        </div>
    </body>
    </html>
    """
    
    text_body = f"""
    灵国国库 - バンクノート生成が可能になりました！
    
    {username}さん、
    
    クールダウン期間が終了し、新しいバンクノートを生成できるようになりました！
    
    今すぐダッシュボードにアクセスして、新しいバンクノートを生成しましょう。
    http://localhost:5000/dashboard
    
    © 2026 灵国国库 | Ling Country Treasury
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
