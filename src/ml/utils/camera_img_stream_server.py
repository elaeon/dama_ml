import io
import socket
import struct
import time
import picamera

#def take_picture():
while True:
    try:
        server_socket = socket.socket()
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind(('0.0.0.0', 8000))
        server_socket.listen(0)
        conn, addr = server_socket.accept()
        num_images = int(conn.recv(20))
        connection = conn.makefile('wb')
        print("Conection accepted, {}".format(addr))
        with picamera.PiCamera() as camera:
            camera.resolution = (640, 480)
            camera.framerate = 30
            camera.rotation = 180
            print("Starting warming")
            time.sleep(1)
            stream = io.BytesIO()
            for i, foo in enumerate(camera.capture_continuous(stream, 'jpeg', use_video_port=True)):
                connection.write(struct.pack('<L', stream.tell()))
                connection.flush()
                # Rewind the stream and send the image data over the wire
                stream.seek(0)
                connection.write(stream.read())
                # Reset the stream for the next capture
                stream.seek(0)
                stream.truncate()
                if i == num_images:
                    break
        # Write a length of zero to the stream to signal we're done
        connection.write(struct.pack('<L', 0))
        connection.close()
    except socket.error:
        print("Error, connection close abnormaly...")
    finally:
        server_socket.close()
        print("Close conections...")

#if __name__ == '__main__':
#    take_picture()
