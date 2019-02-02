import threading
import time


def fun():
    for _ in range(10):
        print("Hello")
        time.sleep(1)




def main():

    threads = []

    for _ in range(5):
        t = threading.Thread(target=fun)
        t.start()
        threads.append(t)


    for t in threads:
        t.join()



if __name__ == "__main__":
    main()




