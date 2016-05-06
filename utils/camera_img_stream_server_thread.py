import io
import socket
import struct
import time
import picamera
import threading

THREADS = 4
#COUNTER = 0
while True:
    try:
        print("Waiting for connection")
        server_socket = socket.socket()
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind(('0.0.0.0', 8000))
        server_socket.listen(0)
        
        conn, addr = server_socket.accept()
        num_images = int(conn.recv(20))
        connection = conn.makefile('wb')
        print("Conection accepted, {}".format(addr))
        
        connection_lock = threading.Lock()
        pool_lock = threading.Lock()
        pool = []

        class ImageStreamer(threading.Thread):
            def __init__(self):
                super(ImageStreamer, self).__init__()
                self.stream = io.BytesIO()
                self.event = threading.Event()
                self.terminated = False
                self.start()
                self.close_abruptly = False

            def run(self):
                # This method runs in a background thread
                #global COUNTER
                while not self.terminated:
                    # Wait for the image to be written to the stream
                    if self.event.wait(1):
                        try:
                            with connection_lock:
                                connection.write(struct.pack('<L', self.stream.tell()))
                                connection.flush()
                                self.stream.seek(0)
                                connection.write(self.stream.read())
                                #COUNTER += 1
                            #if COUNTER >= num_images:
                            #    self.terminated = True
                            #    self.close_abruptly = True
                        except socket.error, AttributeError:
                            print("Error, connection close abnormaly")
                            self.terminated = True
                            self.close_abruptly = True
                        finally:
                            self.stream.seek(0)
                            self.stream.truncate()
                            self.event.clear()
                            with pool_lock:
                                pool.append(self)

        def streams():
            run = True
            TOTAL_TRIES = 30
            tries = TOTAL_TRIES
            while run:
                with pool_lock:
                    if len(pool) > 0:
                        streamer = pool.pop()
                    else:
                        streamer = None
                if streamer:
                    if streamer.close_abruptly:
                        run = False
                    else:
                        yield streamer.stream
                        streamer.event.set()
                        tries = TOTAL_TRIES
                else:
                    if tries == 0:
                        run = False
                    # When the pool is starved, wait a while for it to refill
                    time.sleep(0.1)
                    tries -= 1


        with picamera.PiCamera() as camera:
            pool = [ImageStreamer() for i in range(THREADS)]
            camera.resolution = (640, 480)
            camera.framerate = 30
            camera.rotation = 180
            time.sleep(1)
            camera.capture_sequence(streams(), 'jpeg', use_video_port=True)

        # Shut down the streamers in an orderly fashion
        while pool:
            streamer = pool.pop()
            streamer.terminated = True
            streamer.join()

        # Write the terminating 0-length to the connection to let the server
        # know we're done
        with connection_lock:
            connection.write(struct.pack('<L', 0))

        connection.close()
    except socket.error:
        print("Error, connection close abnormaly...")
        time.sleep(1)
    finally:
        server_socket.close()
        print("Close conections...")
