from socket import *

cliSock = socket(AF_INET, SOCK_DGRAM)
addr = ("localhost", 9000)
while(1):
    data = input(">>>")
    if not data:
        break
    data = data.encode(encoding="utf-8")
    cliSock.sendto(data, addr)
cliSock.close()