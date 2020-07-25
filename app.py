import streamlit as st 
import numpy as np 
import base64 
import matplotlib.pyplot as plt 
import cv2
from PIL import Image 
import io
from utils import *
import webbrowser


def main():
    
    function = ["About","Blurring","Edge Detection","Color Spaces","Fourier Transformation","Histograms","Editing Filters","Object Detection","Segmentation","Image Super Resolution"]
    choice = st.sidebar.selectbox("Menu",function)
    
    if choice=="About" :
        st.title("Auto Image Processing Tool")
        st.subheader("Digital Image Processing")
        intro = "Digital Image Processing means processing digital image by means of a digital computer.\nWe can also say that it is a use of computer algorithms, in order \nto get enhanced image either to extract some useful information."
        st.write(intro)
        file_ = open("media/giphy.gif", "rb")
        contents = file_.read()
        data_url = base64.b64encode(contents).decode("utf-8")
        file_.close()
        st.markdown(
            f'<img src="data:image/gif;base64,{data_url}" alt="cat gif">',
            unsafe_allow_html=True,
        )
        st.subheader("Go around the sidebar on left to explore different functions.")
        
        if st.button("Github"):
            webbrowser.open('https://github.com/harshvardhan33')
        if st.button("Linkedin"):
            webbrowser.open('www.linkedin.com/in/harshvardhan33')
        

    
    if choice == "Blurring":
        st.title("Image Blurring")
        intro = "Blurring of an image means to  reduce the edge content and makes the transition form one color to the other very smooth."
        st.write(intro)
        data = st.file_uploader("Upload the Image :", type=["jpg", "jpeg","png"])
        try:
            if data is not None:
                img = np.array(Image.open(io.BytesIO(data.read()))) 
                grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                st.image(img,width=400)
                st.success("Succesfully Uploaded")
            select = st.radio("What type of image you want to work on!!",("Colored","GrayScale"))
            ksize_sel = st.radio("Specify the kernel Size",("3x3","5x5","7x7","9x9","17x17"))
            if select =="Colored" and img is not None:
                temp_img = img.copy()
            if select=="GrayScale":
                temp_img = grayImage.copy()
            
            if ksize_sel=="3x3" and img is not None:
                ksize = (3,3)
            if ksize_sel=="5x5":
                ksize = (5,5)
            if ksize_sel=="7x7":
                ksize = (7,7)
            if ksize_sel=="9x9":
                ksize = (9,9)
            if ksize_sel=="17x17":
                ksize = (17,17)
            
            if st.checkbox("1. Mean Filter"):
                out = meanFilter(temp_img,ksize)
                st.image(out,width=400)
            if st.checkbox("2. Median Filter"):
                out = medianFilter(temp_img,ksize)
                st.image(out,width=400)
            if st.checkbox("3. Gaussian Filter"):
                out = gaussianFilter(temp_img,ksize)
                st.image(out,width=400)
            if st.checkbox("4. Laplacian Filter"):
                out = laplacianFilter(temp_img,ksize)
                st.image(out,width=400)
        except:
            pass
     
    if choice=="Edge Detection":
        st.title("Edge Detection")
        intro="Sudden changes of discontinuities in an image are called as edges. Significant transitions in an image are called as edges."
        st.write(intro)
        data = st.file_uploader("Upload the Image :", type=["jpg", "jpeg","png"])

     
        if data is not None:
            img = np.array(Image.open(io.BytesIO(data.read()))) 
            grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            st.image(img,width=400)
            st.success("Succesfully Uploaded")
        ksize_sel = st.radio("Specify the kernel Size",("3x3","5x5","7x7","9x9","17x17"))
        try:       
            if ksize_sel=="3x3" and img is not None:
                ksize = (3,3)
            if ksize_sel=="5x5":
                ksize = (5,5)
            if ksize_sel=="7x7":
                ksize = (7,7)
            if ksize_sel=="9x9":
                ksize = (9,9)
            if ksize_sel=="17x17":
                ksize = (17,17)
            if st.checkbox("1. Sobel Edge Detection"):
                out = sobelEdge(img,ksize)
                st.image(out,width=400)
            if st.checkbox("2. Laplacian Edge Detection"):
                out = laplacianEdge(img,ksize)
                st.image(out,width=400)
            if st.checkbox("3. Canny Edge Detection"):
                out = cannyEdge(img,ksize)
                st.image(out,width=400)
        
        except:
            pass
            
    if choice=="Color Spaces":
        st.title("Color Spaces")
        intro="A color space is a specific organization of colors. In combination with physical device profiling, it allows for reproducible representations of color, in both analog and digital representations."
        st.write(intro)
        file_ = open("media/colorSpace.gif", "rb")
        contents = file_.read()
        data_url = base64.b64encode(contents).decode("utf-8")
        file_.close()
        st.markdown(
            f'<img src="data:image/gif;base64,{data_url}" alt="cat gif">',
            unsafe_allow_html=True,
        )
        data = st.file_uploader("Upload the Image :", type=["jpg", "jpeg","png"])
        if data is not None:
            img = np.array(Image.open(io.BytesIO(data.read()))) 
            grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            st.image(img,width=400)
            st.success("Succesfully Uploaded")
        
        st.subheader("OpenCV by default reads images in BGR format")            
    
    
        try:
            if st.checkbox("RGB Space"):
                out= toRGB(img)
                st.image(out,width=400)
            
            if st.checkbox("YCrCb Space"):
                out = toYCrCb(img)
                st.image(out,width=400)
                
            if st.checkbox("HSV Space"):
                out = toHSV(img)
                st.image(out,width=400)
            
            if st.checkbox("LAB Space"):
                out = toLAB(img)
                st.image(out,width=400)
            
            if st.checkbox("Heatmap"):
                  out = toHeatmap(img)
                  st.image(out,width=400)
        except:
            pass
        
    if choice=="Fourier Transformation":
        st.title("Fourier Transformation")
        intro1="The Fourier Transform is an important image processing tool which is used to decompose an image into its sine and cosine components."
        intro2="The output of the transformation represents the image in the Fourier or frequency domain, while the input image is the spatial domain equivalent."
        intro3="The Fourier Transform is used in a wide range of applications, such as image analysis, image filtering, image reconstruction and image compression."
        st.write(intro1,intro2,intro3)
        file_ = open("media/fourier.gif", "rb")
        contents = file_.read()
        data_url = base64.b64encode(contents).decode("utf-8")
        file_.close()
        st.markdown(
            f'<img src="data:image/gif;base64,{data_url}" alt="cat gif">',
            unsafe_allow_html=True,
        )
        data = st.file_uploader("Upload the Image :", type=["jpg", "jpeg","png"])
        if data is not None:
            img = np.array(Image.open(io.BytesIO(data.read()))) 
            grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            st.image(img,width=400)
            st.success("Succesfully Uploaded")
        
        try:
        
            if st.checkbox("Compute FFT"):           
                   
                st.header("Spectrum")
                original = np.fft.fft2(grayimg)
                plt.imshow(np.log(1+np.abs(original)), "gray")
                st.pyplot()
           
                st.header("Centered Spectrum")
                center = np.fft.fftshift(original)
                plt.imshow(np.log(1+np.abs(center)), "gray")
                st.pyplot()
             
                st.header("Decentralized")
                inv_center = np.fft.ifftshift(center)
                plt.imshow(np.log(1+np.abs(inv_center)), "gray")
                st.pyplot()
                
                st.header("Phase Angle")
                phase = np.fft.fft2(grayimg)
                plt.imshow(np.angle(phase))
                st.pyplot()
                
                
                st.header("Processed Image")
                processed_img = np.fft.ifft2(inv_center)
                plt.imshow(np.abs(processed_img), "gray")
                st.pyplot()
                
                
            if st.checkbox("Low Pass Filter"):
                
                
                original = np.fft.fft2(grayimg)
                center = np.fft.fftshift(original)
       
                LowPass = idealFilterLP(50,img.shape)
                plt.imshow(np.abs(LowPass), "gray")
                
                st.header("Centered Spectrum multiply Low Pass Filter")
                LowPassCenter = center * idealFilterLP(50,grayimg.shape)
                plt.imshow(np.log(1+np.abs(LowPassCenter)), "gray")
                st.pyplot()           
                
                st.header("Decentralize")
                LowPass = np.fft.ifftshift(LowPassCenter)
                plt.imshow(np.log(1+np.abs(LowPass)), "gray")
                st.pyplot()           
                
                st.header("Processed Image")           
                inverse_LowPass = np.fft.ifft2(LowPass)
                plt.imshow(np.abs(inverse_LowPass), "gray")
                st.pyplot()
            
            
            if st.checkbox("High Pass Filter"):
                original = np.fft.fft2(grayimg)
                center = np.fft.fftshift(original)
                
                st.header("High Pass Filter")        
                HighPass = idealFilterHP(50,img.shape)
                plt.imshow(np.abs(HighPass), "gray")
                st.pyplot()
                
                
                st.header("Centered Spectrum multiply High Pass Filter")
                HighPassCenter = center * idealFilterHP(50,img.shape)
                plt.imshow(np.log(1+np.abs(HighPassCenter)), "gray")
                st.pyplot()
                
                st.header("Decentralize")
                HighPass = np.fft.ifftshift(HighPassCenter)
                plt.imshow(np.log(1+np.abs(HighPass)), "gray")
                st.pyplot()
                
                st.header("Processed Image")
                inverse_HighPass = np.fft.ifft2(HighPass)
                plt.imshow(np.abs(inverse_HighPass), "gray")
                st.pyplot()
            
            if st.checkbox("Ideal Filter"):
                
                st.header("Ideal Low Pass Filter")
                LowPass = idealFilterLP(50,img.shape)
                plt.imshow(LowPass, "gray")
                st.pyplot()
                
                
                
                
                st.header("Ideal High Pass Filter")
                HighPass = idealFilterHP(50,img.shape)
                plt.imshow(HighPass, "gray")
                st.pyplot()
                plt.show()
            
            if st.checkbox("Butterworth Filter"):
                st.header("Low Pass Butterworth Filter")
                LowPass = butterworthLP(50,img.shape,20)
                plt.imshow(LowPass, "gray")
                st.pyplot()
                
                st.header("High Pass Butterworth Filter")
                HighPass = butterworthHP(50,img.shape,20)
                plt.imshow(HighPass, "gray")
                st.pyplot()
            
            if st.checkbox("Gaussian Filter"):
                
                st.header("Low Pass Gaussian Filter")
                LowPass = gaussianLP(50,img.shape)
                plt.imshow(LowPass, "gray")
                st.pyplot()
                
                st.header("High Pass Gaussian Filter")
                HighPass = gaussianHP(50,img.shape)
                plt.imshow(HighPass, "gray")
                plt.show()
                st.pyplot()
        except:
            pass
        
    if choice=="Histograms":
        st.title("Histogram")
        intro = "A histogram is a very important tool in Image processing. It is a graphical representation of the distribution of data. An image histogram gives a graphical representation of the distribution of pixel intensities in a digital image."
        st.write(intro)
        file_ = open("media/hist.gif", "rb")
        contents = file_.read()
        data_url = base64.b64encode(contents).decode("utf-8")
        file_.close()
        st.markdown(
            f'<img src="data:image/gif;base64,{data_url}" alt="cat gif">',
            unsafe_allow_html=True,
        )
        data = st.file_uploader("Upload the Image :", type=["jpg", "jpeg","png"])

     
        if data is not None:
            img = np.array(Image.open(io.BytesIO(data.read()))) 
            grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            st.image(img,width=400)
            st.success("Succesfully Uploaded")
        
        try:            
            select = st.radio("What type of image you want to work on!!",("Colored","GrayScale"))
            if select =="Colored" and img is not None:
                temp_img = img.copy()
            if select=="GrayScale":
                temp_img = grayImage.copy()
                
            
            if st.checkbox("Compute Histogram"):
                if select =="Colored":
                    st.title("Histogram of the Pixel Intensities in RGB")
                    for i, col in enumerate(['b', 'g', 'r']):
                        hist = cv2.calcHist([img], [i], None, [256], [0, 256])
                        plt.plot(hist, color = col)
                        plt.xlim([0, 256])      
                    plt.show()
                    st.pyplot()
                                    
                if select=="GrayScale":
                    st.title("Histogram of the Pixel Intensities in GrayScale")
                    histogram = cv2.calcHist([temp_img], [0], None, [256], [0, 256])
                    plt.plot(histogram, color='k')
                    plt.show()
                    st.pyplot()
            
            if st.checkbox("Histogram Equalization"):
                if select =="Colored":
                    channels = cv2.split(temp_img)
                    eq_channels = []
                    for ch, color in zip(channels, ['B', 'G', 'R']):
                        eq_channels.append(cv2.equalizeHist(ch))
                
                    eq_image = cv2.merge(eq_channels)
                    eq_image = cv2.cvtColor(eq_image, cv2.COLOR_BGR2RGB)
                    plt.imshow(eq_image)
                    plt.show()
                    st.pyplot()
                    
                    
                    for i, col in enumerate(['b', 'g', 'r']):
                        hist = cv2.calcHist([eq_image], [i], None, [256], [0, 256])
                        plt.plot(hist, color = col)
                        plt.xlim([0, 256])      
                    plt.show()
                    st.pyplot()
                    
                                                
                if select=="GrayScale":
                    st.title("Image After Histogram Equalization")
                    eq_grayscale_image = cv2.equalizeHist(temp_img)
                    plt.imshow(eq_grayscale_image, cmap='gray')
                    plt.show()
                    st.pyplot()
                    
                    st.title("New Histogram")
                    histogram = cv2.calcHist([eq_grayscale_image], [0], None, [256], [0, 256])
                    plt.plot(histogram, color='k')
                    plt.show()
                    st.pyplot()
        except:
            pass
        
    if choice=="Editing Filters":
        st.title("Famous Editing Filters")
        intro = "You can play around with some famous editing filters you come across in many softwares."
        st.write(intro)
        data = st.file_uploader("Upload the Image :", type=["jpg", "jpeg","png"])
        
        if data is not None:
            img = np.array(Image.open(io.BytesIO(data.read()))) 
            grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            st.image(img,width=400)
            st.success("Succesfully Uploaded")
            
        try:            
            if st.checkbox("Brighten"):
                new_image = brighten(img)
                st.image(new_image,width=400)
            if st.checkbox("Enhance"):
                new_image1,new_image2 = enhance(img)
                st.image([new_image1,new_image2],width=325)
            if st.checkbox("Inversion"):
                new_image = inversion(img)
                st.image(new_image,width=400)
            if st.checkbox("Cold"):
                new_image = cold(img)
                st.image(new_image,width=400)
            if st.checkbox("Warm"):
                new_image = warm(img)
                st.image(new_image,width=400)
    
            if st.checkbox("Cartoon"):
                new_image = cartoon(img)
                st.image(new_image,width=400)
            if st.checkbox("Sketching"):
                new_image1,new_image2 = sketch(img)
                st.image([new_image1,new_image2],width=325)
        except:
            pass

    if choice=="Object Detection":
        st.header("Stay Tuned!!, Coming Soon")
    if choice=="Segmentation":
        st.header("Stay Tuned!!, Coming Soon")
    if choice=="Image Super Resolution":
        st.header("Stay Tuned!!, Coming Soon")


if __name__ == '__main__':
	main()