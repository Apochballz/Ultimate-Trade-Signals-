#!/usr/bin/env python3
"""
Flask wrapper for Forex Signal Bot - For Render deployment
This file creates a web server that runs the bot in the background.
"""

import os
import threading
import asyncio
from flask import Flask, jsonify
import sys

# Import the main function from your bot
from streamlined_forex_bot import main as start_bot_main

app = Flask(__name__)

# Global variable to track if bot is running
bot_thread = None
bot_running = False

@app.route('/')
def home():
    """Home endpoint - shows bot status"""
    return jsonify({
        'status': 'running' if bot_running else 'starting',
        'service': 'Forex Signal Bot',
        'message': 'Bot is running in background'
    })

@app.route('/health')
def health():
    """Health check endpoint for Render"""
    return jsonify({'status': 'healthy'}), 200

@app.route('/start-bot')
def start_bot():
    """Endpoint to manually start the bot"""
    global bot_thread, bot_running
    
    if bot_running:
        return jsonify({'status': 'already_running'}), 200
    
    def run_bot():
        global bot_running
        try:
            bot_running = True
            print("Starting bot...")
            # Run the bot's main function
            start_bot_main()
        except Exception as e:
            print(f"Bot error: {e}")
            bot_running = False
    
    # Start bot in a separate thread
    bot_thread = threading.Thread(target=run_bot, daemon=True)
    bot_thread.start()
    
    return jsonify({'status': 'started'}), 200

@app.route('/stop-bot')
def stop_bot():
    """Endpoint to stop the bot (for debugging)"""
    global bot_running
    
    # Note: This is a simple implementation
    # For proper shutdown, you'd need to implement signal handling
    bot_running = False
    return jsonify({'status': 'stopping'}), 200

def start_background_bot():
    """Start the bot automatically when Flask app starts"""
    global bot_thread, bot_running
    
    print("Initializing background bot...")
    
    def run_bot():
        global bot_running
        try:
            bot_running = True
            print("🚀 Starting Forex Signal Bot in background thread...")
            # Run the bot's main function
            start_bot_main()
        except Exception as e:
            print(f"Bot error: {e}")
            bot_running = False
            # Optionally restart after delay
            import time
            time.sleep(10)
            if not bot_running:
                print("Attempting to restart bot...")
                run_bot()
    
    # Start bot in background thread
    bot_thread = threading.Thread(target=run_bot, daemon=True)
    bot_thread.start()
    print("✅ Bot started in background thread")

# Start bot when module loads
if os.environ.get('RENDER', ''):
    print("Running on Render, starting bot automatically...")
    start_background_bot()

if __name__ == '__main__':
    # Start bot when running locally
    start_background_bot()
    
    # Get port from environment variable (Render sets this)
    port = int(os.environ.get('PORT', 10000))
    
    # Start Flask app
    print(f"Starting Flask server on port {port}...")
    app.run(host='0.0.0.0', port=port, debug=False)