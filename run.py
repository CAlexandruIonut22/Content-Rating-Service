from app import app

if __name__ == '__main__':
    print("Pornirea aplicației Content Rating Service...")
    app.run(debug=True, host='127.0.0.1', port=5000)