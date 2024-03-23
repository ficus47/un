import socket as sk

x = sk.gethostbyname(sk.gethostname())
print(x)

sock = sk.socket()
sock.bind((x, 8080))

sock.listen(5)

with open("video.mp4", "wb") as f:
  while True:
    con, adr = sock.accept()
    a = con.recv(10240)
    print(adr)
    f.write(a)
