import cv2
from os import listdir,mkdir
from os.path import isfile, join
from tkinter import *
from tkinter import messagebox
from tkinter.font import  Font
import pyttsx3 as t2s
import threading
from pyautogui import *
import wmi
from pandas import *
from PIL import ImageTk, Image



ct=0
brightness=50
convert_constant = 1
inverse_convert_constant = 1

face_cascade = cv2.CascadeClassifier('face_cas.xml')

def MAin():
    root_un=Tk()
    root.destroy()
    l=Label(root_un,text="Done Login").pack()
    root_un.mainloop()


def face_detect(img):
    face_cascade = cv2.CascadeClassifier('face_cas.xml')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
    n = 0
    for (xx, yy, ww, hh) in face:
        cv2.rectangle(img, (xx, yy), (xx + ww, yy + hh), (255, 255, 0), 2)
        n = face.shape[0]
    return n

def Signup():
    L1 = Label(root, text='                                                                           ', font=f3).place(x=300, y=420)
    try:
        def sign_up():
            sig = sign.get()
            Label(root_signup, text='                                   ').place(x=200, y=180)
            if sig == 'Kundan':
                x = os.listdir('D:/Face UnLock Face_Data')
                if len(x) == 0:
                    pass
                else:
                    for i in x:
                        os.remove('D:/Face UnLock Face_Data/%s' % i)

                data_path = 'D:/Face UnLock Face_Data/'
                onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path, f))]
                Labels = []
                for i, files in enumerate(onlyfiles):
                    Labels.append(i)
                Labels = np.asarray(Labels, dtype=np.int32)

                cap = cv2.VideoCapture(0)
                count = 0
                x1 = len(Labels)
                print(x1)

                while True:
                    ret, frame = cap.read()
                    frame = cv2.flip(frame, 1)
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                    n = 0
                    for (x, y, w, h) in faces:
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
                        n = faces.shape[0]
                        face_img = frame[y:y + h, x:x + w]

                    if n == 0:
                        cv2.putText(frame, 'No Face found', (10, 450), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0), 2)
                    else:

                        count += 1
                        face = cv2.resize(face_img, (200, 200))
                        face =gray

                        cv2.imwrite('D:/Face UnLock Face_Data/user' + str(count + x1) + '.jpg', face)

                        cv2.putText(face, str(count), (5, 180), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                        cv2.imshow('Face', face)

                    if cv2.waitKey(1) == ord('q') or count == 100:
                        break
                count = 0
                cap.release()
                cv2.destroyAllWindows()
                root_signup.destroy()
            else:
                Label(root_signup, text='Invalid Password',font=f3).place(x=100, y=350)

        root_signup = Toplevel()
        root_signup.geometry('360x500+900+100')
        img1 = ImageTk.PhotoImage(Image.open("12_lock.jpg"))
        panel = Label(root_signup, image=img1).place(x=1, y=20)
        sign = StringVar()
        l3 = Label(root_signup, text="Machine Learning ", fg='brown', font=f1).place(x=10, y=10)
        l3 = Label(root_signup, text=" Sign Up with Face_Lock", fg='darkblue', font=f1).place(x=60, y=40)
        l3 = Label(root_signup, text=" Enter Password ", fg='blue', font=f1).place(x=100, y=380)
        E1 = Entry(root_signup, show='*', textvariable=sign,font=f3).place(x=90, y=420)
        but = Button(root_signup, text='Login', command=sign_up, width=20, height=1, font=f3, bg='green').place(x=80,
                                                                                                                y=460)
        root_signup.resizable('false', 'false')
        root_signup.mainloop()
    except:
        pass

def Face_Unlock():
    try:

        data_path = 'D:/Face UnLock Face_Data/'
        onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path, f))]

        Training_Data, Labels = [], []

        for i, files in enumerate(onlyfiles):
            image_path = data_path + onlyfiles[i]
            images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            Training_Data.append(np.asarray(images, dtype=np.uint8))
            Labels.append(i)
        if len(Labels) == 0:
            eng = t2s.init()
            try:
                eng.setProperty('rate', 140);
                eng.setProperty('volume', .9)
                eng.say('Firstly You should SignUp With Face ID')
                eng.runAndWait()
            except:
                pass

            o1 = "Message: Firstly You should SignUp With Face ID"
            L1 = Label(root, text=o1, font=f3).place(x=300, y=420)

        else:
            Labels = np.asarray(Labels, dtype=np.int32)

            model = cv2.face.LBPHFaceRecognizer_create()

            model.train(np.asarray(Training_Data), np.asarray(Labels))

            ct = 0
            cap = cv2.VideoCapture(1)
            while True:

                ret, frame = cap.read()
                frame = cv2.flip(frame, 1)
                n = 0

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.3, 5)

                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
                    face = gray[y:y + h, x:x + w]
                    face = cv2.resize(face, (200, 200))
                    n = face.shape[0]
                    j, k = model.predict(face)
                    confidence = int(100 * (1 - (k) / 300))

                if n > 0:

                    if k < 50:
                        if confidence > 75:
                            ct += 1
                            cv2.putText(frame, '% Matching' + str(confidence), (20, 450), cv2.FONT_HERSHEY_COMPLEX, 1,
                                        (0, 0, 255), 1)

                            cv2.putText(frame, 'unlocking', (420, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 125), 3)

                        else:
                            ct = 0
                            cv2.putText(frame, "Unknown", (x + w, y + h + 10), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0),
                                        1)
                    else:
                        ct = 0
                        cv2.putText(frame, "Unknown", (x + w, y + h + 10), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)

                if n == 0:
                    cv2.putText(frame, "Face Not Found", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)

                cv2.imshow('Face unLockIng Application', frame)

                if cv2.waitKey(1) == ord('q') or ct == 20:
                    break

            cap.release()
            cv2.destroyAllWindows()

            if ct == 20:
                eng = t2s.init()
                try:
                    eng.setProperty('rate', 140)
                    eng.setProperty('volume', .9)
                    eng.say('login successfuly. Loading the Content')
                    eng.runAndWait()
                except:
                    pass

                MAin()
    except:
        pass

def login():
    s1 = ss1.get()
    s2 = ss2.get()
    if (s1 == 'Kundan' and s2 == 'Singh'):
        ss1.set('')
        ss2.set('')
        eng = t2s.init()
        try:
            eng.setProperty('rate', 140);
            eng.setProperty('volume', .9)
            eng.say('login successfuly. Loading the Content')
            eng.runAndWait()
        except:
            pass
        MAin()
    elif s1 == '' and s2 == '':

        eng = t2s.init()
        try:
            eng.setProperty('rate', 140);
            eng.setProperty('volume', .9)
            eng.say('First You should Enter Username and Password.')
            eng.runAndWait()
        except:
            pass

        o1 = "Message: Please Enter Username and Password"
        L1 = Label(root, text=o1, font=f3).place(x=300, y=420)

    else:
        eng = t2s.init()
        ss2.set('')
        try:
            eng.setProperty('rate', 140);
            eng.setProperty('volume', .9)
            eng.say('You have Entered Wrong username or password')
            eng.runAndWait()
        except:
            pass

        o2 = "Please Enter Invalid Username Or Password"
        L2 = Label(root, text=o2, font=f3).place(x=300, y=420)

def win():
    ans1 = messagebox.askyesno("Exit", "DO You Want to Exit")
    if ans1 == True:
        root.destroy()


try:
    mkdir('D:/Face UnLock Face_Data/')
except:
    pass

root=Tk()

frame=Frame(root,height=600,width=10).pack(side=LEFT)
img = ImageTk.PhotoImage(Image.open("1.jpg"))
panel = Label(frame, image = img)
panel.pack(side = "right", fill = "both", expand = "yes")

f1 = Font(family="Time New Roman", size=12, weight="bold", underline=1)
f2 = Font(family="Time New Roman", size=15, weight="bold", underline=1)
f3 = Font(family="Time New Roman", size=10, weight="bold")

eng = t2s.init()

def text2speech():
    try:
        eng.setProperty('rate', 140);
        eng.setProperty('volume', .9)
        eng.say('Welcome To Machine Learning Training Project. its multifunction window application. Enter Username and password to continue or you can use face unlock feature')
        eng.runAndWait()
    except:
        pass


root.title("Training Project")
l3 = Label(root, text="Machine Learning ", fg='brown', font=f1,bg='white').place(x=150, y=30)
l3 = Label(root, text=" Virtual Application", fg='green', font=f1,bg='white').place(x=100, y=70)
l3 = Label(root, text="Enter Username and Password", fg='brown', font=f1,bg='white').place(x=50, y=120)
l3 = Label(root, text="Copyright @ Kundan Kumar ", fg='skyblue',bg='white').place(x=550, y=420)
l1 = Label(root, text='Username', fg='brown', font=f1,bg='white').place(x=50, y=160)
l2 = Label(root, text='Password', fg='brown', font=f1,bg='white').place(x=50, y=200)

ss1 = StringVar()
ss2 = StringVar()
e1 = Entry(root, textvariable=ss1).place(x=205, y=165)
e2 = Entry(root, textvariable=ss2, show='*').place(x=205, y=205)
b1 = Button(root, text='Login', command=login, width=15, height=1, bg='skyblue', fg='white', font=f3).place(x=50,y=250)
b2 = Button(root, text='Login_With_Face', command=Face_Unlock, width=15, height=1, bg='green', fg='white', font=f3).place(x=210, y=250)
b1 = Button(root, text='Sign_up',width=31,command=Signup,bg='orange',font=f3,height=1).place(x=50,y=290)
b3 = Button(root, text='Exit', command=win, width=31, height=1, font=f3).place(x=50, y=330)

c1 = Canvas(root, width=20, height=400, bg='White')
c1.pack(side=RIGHT)
c1 = Canvas(root, width=10, height=400, bg='orange')
c1.pack(side=RIGHT)
root.geometry("800x450+500+100")

try:
    t = threading.Thread(name='child', target=text2speech, args=())
    if not t.is_alive():
        t.start()
except:
    pass

root.resizable('false','false')
root.mainloop()
