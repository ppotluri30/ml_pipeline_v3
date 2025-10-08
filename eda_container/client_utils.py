# client_utils.py
import requests # type: ignore
import io

def post_file(fastapi_url: str, bucket_name: str, object_name: str, file_content):
    """
    Sends data held in a Python variable to the FastAPI upload endpoint.
    """

    print(f"Preparing to upload {len(file_content)} bytes of data.")

    # Construct the full URL for the upload endpoint.
    url = f"{fastapi_url}/upload/{bucket_name}/{object_name}"
    
    # Set the content type header.
    # This tells the server what kind of data you are sending.
    headers = {
        'Content-Type': 'application/octet-stream'
    }

    print(f"Sending POST request to: {url}")
    try:
        # Send the POST request.
        # The 'data' parameter takes the bytes from our variable and places them in the request body.
        response = requests.post(url, data=file_content, headers=headers)

        # Raise an exception for bad status codes (4xx or 5xx)
        response.raise_for_status()

        # Print the success response from the server
        print("Upload successful!")
        print("Server response:", response.json())
        return True

    except requests.exceptions.RequestException as e:
        print(f"An error occurred during upload: {e}")
        if e.response:
            print("Error details:", e.response.text)

def get_file(fastapi_url: str, bucket_name: str, object_name: str):
    """
    Connects to the FastAPI endpoint to stream a file's content
    and store it in a variable.
    """

    # Construct the full URL for the API endpoint
    url = f"{fastapi_url}/download/{bucket_name}/{object_name}"
    print(f"Attempting to get file content from: {url}")

    try:
        # 'stream=True' us to process the response as it arrives, chunk by chunk.
        with requests.get(url, stream=True) as r:
            # Raise an exception for bad status codes (4xx or 5xx)
            r.raise_for_status()

            print("Successfully connected. Streaming content into variable...")
            
            # Create an in-memory binary stream.
            file_stream = io.BytesIO()

            # Download the file chunk by chunk and write to the in-memory stream.
            for chunk in r.iter_content(chunk_size=8192):
                file_stream.write(chunk)

            # IMPORTANT: Rewind the stream to the beginning before it's used.
            file_stream.seek(0)

            print("Finished streaming. File content is now in memory.")
            print(f"Total size of content: {file_stream.getbuffer().nbytes} bytes")

            return file_stream

    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        return None
