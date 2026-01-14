# Email management routes for profile
from flask import request, session, jsonify, redirect, url_for, flash
from models import db, User, EmailHistory

def register_email_routes(app):
    """Register email management routes with the Flask app"""
    
    @app.route("/profile/change-email", methods=["POST"])
    def change_email():
        """Initiate email change process"""
        if "user_id" not in session:
            return jsonify({"success": False, "error": "Not logged in"}), 401
        
        user = User.query.get(session["user_id"])
        if not user:
            return jsonify({"success": False, "error": "User not found"}), 404
        
        new_email = request.form.get("new_email", "").strip()
        password = request.form.get("password", "")
        
        # Validate password
        if not user.check_password(password):
            return jsonify({"success": False, "error": "Incorrect password"}), 400
        
        # Validate email format
        if not new_email or '@' not in new_email:
            return jsonify({"success": False, "error": "Invalid email address"}), 400
        
        # Check if email is already in use
        existing_user = User.query.filter_by(email=new_email).first()
        if existing_user and existing_user.id != user.id:
            return jsonify({"success": False, "error": "Email already in use"}), 400
        
        # Same email as current
        if new_email == user.email:
            return jsonify({"success": False, "error": "This is already your current email"}), 400
        
        # Store pending email and generate new verification token
        user.pending_email = new_email
        user.email_verified = False  # Require re-verification
        verification_token = user.generate_verification_token()
        db.session.commit()
        
        # Send verification email to NEW email address
        try:
            from email_service import send_email_change_verification
            app_url = request.url_root.rstrip('/')
            send_email_change_verification(new_email, user.username, verification_token, user.email, app_url)
            return jsonify({"success": True, "message": "Verification email sent to new address. Please check your inbox."})
        except Exception as e:
            print(f"[ERROR] Failed to send email change verification: {e}")
            return jsonify({"success": False, "error": "Failed to send verification email"}), 500
    
    
    @app.route("/profile/verify-email-change/<token>")
    def verify_email_change(token):
        """Complete email change after verification"""
        if "user_id" not in session:
            flash("Please log in first", "error")
            return redirect(url_for("login"))
        
        user = User.query.get(session["user_id"])
        if not user:
            flash("User not found", "error")
            return redirect(url_for("login"))
        
        if not user.pending_email:
            flash("No pending email change", "info")
            return redirect(url_for("profile", username=user.username))
        
        # Verify token
        if user.verify_email_token(token):
            # Record email change in history
            email_history = EmailHistory(
                user_id=user.id,
                old_email=user.email,
                new_email=user.pending_email,
                ip_address=request.remote_addr,
                user_agent=request.headers.get('User-Agent', '')[:255]
            )
            db.session.add(email_history)
            
            # Update email
            old_email = user.email
            user.email = user.pending_email
            user.pending_email = None
            user.email_verified = True
            db.session.commit()
            
            flash(f"Email successfully changed from {old_email} to {user.email}", "success")
        else:
            flash("Invalid or expired verification token", "error")
        
        return redirect(url_for("profile", username=user.username))
    
    
    @app.route("/profile/cancel-email-change")
    def cancel_email_change():
        """Cancel pending email change"""
        if "user_id" not in session:
            return jsonify({"success": False, "error": "Not logged in"}), 401
        
        user = User.query.get(session["user_id"])
        if not user:
            return jsonify({"success": False, "error": "User not found"}), 404
        
        user.pending_email = None
        user.verification_token = None
        user.verification_token_expires = None
        db.session.commit()
        
        return jsonify({"success": True, "message": "Email change cancelled"})
    
    
    @app.route("/profile/resend-verification")
    def profile_resend_verification():
        """Resend verification email from profile"""
        if "user_id" not in session:
            return jsonify({"success": False, "error": "Not logged in"}), 401
        
        user = User.query.get(session["user_id"])
        if not user:
            return jsonify({"success": False, "error": "User not found"}), 404
        
        if user.email_verified and not user.pending_email:
            return jsonify({"success": False, "error": "Email already verified"}), 400
        
        # Generate new token
        verification_token = user.generate_verification_token()
        db.session.commit()
        
        # Send to pending_email if exists, otherwise current email
        target_email = user.pending_email if user.pending_email else user.email
        
        try:
            from email_service import send_verification_email, send_email_change_verification
            app_url = request.url_root.rstrip('/')
            
            if user.pending_email:
                send_email_change_verification(target_email, user.username, verification_token, user.email, app_url)
            else:
                send_verification_email(target_email, user.username, verification_token, app_url)
            
            return jsonify({"success": True, "message": f"Verification email sent to {target_email}"})
        except Exception as e:
            print(f"[ERROR] Failed to send verification email: {e}")
            return jsonify({"success": False, "error": "Failed to send verification email"}), 500
    
    
    @app.route("/profile/email-history")
    def email_history():
        """View email change history"""
        if "user_id" not in session:
            return jsonify({"success": False, "error": "Not logged in"}), 401
        
        user = User.query.get(session["user_id"])
        if not user:
            return jsonify({"success": False, "error": "User not found"}), 404
        
        history = EmailHistory.query.filter_by(user_id=user.id).order_by(EmailHistory.changed_at.desc()).all()
        
        return jsonify({
            "success": True,
            "history": [{
                "old_email": h.old_email,
                "new_email": h.new_email,
                "changed_at": h.changed_at.strftime("%Y-%m-%d %H:%M:%S"),
                "ip_address": h.ip_address
            } for h in history]
        })
