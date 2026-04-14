import requests

URL = "http://localhost:8000/api/diagnosis/analyze"

def test_text_only():
    print("\n--- Testing Text Only ---")
    data = {"description": "Tôi bị nổi mề đay ngứa quá"}
    response = requests.post(URL, data=data)
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        print(f"Result: {response.json().get('disease_name')} (Conf: {response.json().get('confidence')})")
    else:
        print(response.text)

def test_combined():
    # Giả định có 1 ảnh temp để test
    print("\n--- Testing Combined (Mocking Image + Real Text) ---")
    # Tạo 1 ảnh trắng nhỏ để test
    from PIL import Image
    import io
    img = Image.new('RGB', (100, 100), color = 'red')
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    img_byte_arr = img_byte_arr.getvalue()

    files = {'file': ('test.jpg', img_byte_arr, 'image/jpeg')}
    data = {'description': 'Vảy nến trên da đầu'}
    
    response = requests.post(URL, files=files, data=data)
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        print(f"Result: {response.json().get('disease_name')} (Conf: {response.json().get('confidence')})")
    else:
        print(response.text)

if __name__ == "__main__":
    try:
        test_text_only()
        test_combined()
    except Exception as e:
        print(f"Error connecting to server: {e}")
