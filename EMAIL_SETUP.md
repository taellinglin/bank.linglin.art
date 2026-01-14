# Email System Setup Guide

## メール認証とバンクノート生成通知システムのセットアップガイド

このガイドでは、灵国国库にメール認証とバンクノート生成通知機能を追加する方法を説明します。

## 📋 概要

以下の機能が追加されました:
- ✉️ **メールアドレス認証** - 新規登録時にメール認証が必要
- 🔔 **バンクノート生成通知** - バンクノート生成完了時にメール通知
- 🔄 **認証メール再送** - 認証メールの再送信機能
- 📧 **美しいHTMLメール** - グラデーションデザインのメールテンプレート

## 🚀 セットアップ手順

### 1. 必要なパッケージをインストール

```bash
pip install flask-mail python-dotenv
```

または

```bash
pip install -r requirements_email.txt
```

### 2. 環境変数の設定

`.env`ファイルに以下の設定を追加（既に作成済み）:

```env
MAIL_SERVER=smtp.gmail.com
MAIL_PORT=587
MAIL_USE_TLS=true
MAIL_USERNAME=taellinglin@gmail.com
MAIL_PASSWORD=inlazqvbcvivxxdt
MAIL_DEFAULT_SENDER=taellinglin@gmail.com
```

**注意**: Gmailの場合、アプリパスワードを使用してください。
[Google アプリパスワードの作成方法](https://support.google.com/accounts/answer/185833)

### 3. データベースのマイグレーション

既存のデータベースに新しいカラムを追加します:

```bash
python migrate_email_verification.py
```

これにより、`users`テーブルに以下のカラムが追加されます:
- `email_verified` (Boolean)
- `verification_token` (String)
- `verification_token_expires` (DateTime)

### 4. アプリケーションの起動

```bash
python app.py
```

## 📝 新機能の使い方

### メールアドレス認証

1. **新規登録時**
   - ユーザーが登録すると、自動的に認証メールが送信されます
   - メール内のリンクをクリックして認証完了

2. **認証メールの再送信**
   - `/resend-verification` にアクセス
   - または、ダッシュボードから再送信ボタンをクリック

3. **認証の確認**
   ```python
   user = User.query.filter_by(username='username').first()
   print(user.email_verified)  # True or False
   ```

### バンクノート生成通知

バンクノートが生成されると、以下の情報を含むメールが自動送信されます:
- 生成された枚数
- 額面
- 生成日時
- シリアル番号（最初の5つ）
- ダッシュボードへのリンク

**条件**: メールアドレスが認証済み（`email_verified=True`）のユーザーのみ

## 🔧 トラブルシューティング

### メールが送信されない場合

1. **環境変数を確認**
   ```bash
   python -c "import os; from dotenv import load_dotenv; load_dotenv(); print(os.getenv('MAIL_USERNAME'))"
   ```

2. **Gmailの設定を確認**
   - アプリパスワードを使用しているか
   - 2段階認証が有効になっているか

3. **ファイアウォールの確認**
   - ポート587（TLS）が開いているか

4. **手動テスト**
   ```python
   from app import app
   from email_service import send_verification_email
   
   with app.app_context():
       send_verification_email(
           "test@example.com",
           "TestUser",
           "test_token_123",
           "http://localhost:5000"
       )
   ```

### データベースマイグレーションエラー

カラムが既に存在する場合:
```bash
# エラーを無視して続行（スクリプトが自動処理）
python migrate_email_verification.py
```

手動で確認:
```python
from app import app, db
from models import User

with app.app_context():
    user = User.query.first()
    print(user.email_verified)
```

## 🎨 カスタマイズ

### メールテンプレートの変更

`email_service.py`を編集してHTMLテンプレートをカスタマイズ:

```python
def send_verification_email(user_email, username, verification_token, app_url):
    # html_body を編集
    html_body = f"""
    <!-- Your custom HTML here -->
    """
```

### 通知のタイミング変更

`generate.py`の最後で送信タイミングを調整:

```python
# 生成完了直後に送信（現在の設定）
if svg_pairs_created > 0:
    send_email_notification(...)
```

## 📊 管理者向け機能

### 既存ユーザーのメール認証を手動で承認

```python
from app import app, db
from models import User

with app.app_context():
    user = User.query.filter_by(username='username').first()
    user.email_verified = True
    db.session.commit()
    print(f"✅ {user.username} のメールを認証しました")
```

### 全ユーザーの認証状態を確認

```python
from app import app, db
from models import User

with app.app_context():
    users = User.query.all()
    for user in users:
        status = "✅ 認証済み" if user.email_verified else "❌ 未認証"
        print(f"{user.username}: {status}")
```

## 📁 追加されたファイル

- `.env` - メール設定
- `email_service.py` - メール送信機能
- `migrate_email_verification.py` - DBマイグレーション
- `requirements_email.txt` - 必要なパッケージ
- `EMAIL_SETUP.md` - このガイド

## 🔐 セキュリティ

- メールパスワードは`.env`ファイルに保存（Gitにコミットしない）
- `.gitignore`に`.env`を追加:
  ```
  .env
  ```

- 本番環境では環境変数を使用:
  ```bash
  export MAIL_USERNAME=your-email@gmail.com
  export MAIL_PASSWORD=your-app-password
  ```

## 🚀 本番環境での設定

1. **環境変数を設定**
   ```bash
   # Heroku
   heroku config:set MAIL_USERNAME=your-email@gmail.com
   heroku config:set MAIL_PASSWORD=your-app-password
   
   # AWS/Linux
   export MAIL_USERNAME=your-email@gmail.com
   export MAIL_PASSWORD=your-app-password
   ```

2. **URLを変更**
   `email_service.py`内のハードコードされた`http://localhost:5000`を実際のドメインに変更

3. **HTTPS必須**
   本番環境ではHTTPSを使用してください

## 📞 サポート

問題が発生した場合:
1. ログを確認: `[ERROR]`で検索
2. データベースの状態を確認
3. メール設定を再確認

---

**更新日**: 2026年1月6日
**バージョン**: 1.0
