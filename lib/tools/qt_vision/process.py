import time
from multiprocessing import Process,Queue

MSG_QUEUE = Queue(200)

class MyData(object):
    def __init__(self,data):
        self.data=data

    def use(self):
        print self.data

def startA(msgQueue):
    while True:
        if msgQueue.empty() > 0:
            pass
            # print 'queue is empty %d' % (msgQueue.qsize())
        else:
            msg = msgQueue.get()
            print 'processA : get msg %s' % (msg,)
        time.sleep(1)

def startB(msgQueue):
    while True:
        msgQueue.put('hello world')
        print 'processB: put hello world queue size is %d' % (msgQueue.qsize(),)
        time.sleep(3)


if __name__ == '__main__':
    processA = Process(target=startA,args=(MSG_QUEUE,))
    processB = Process(target=startB,args=(MSG_QUEUE,))

    processA.start()
    print 'processA start..'

    processB.start()
    print 'processB start..'