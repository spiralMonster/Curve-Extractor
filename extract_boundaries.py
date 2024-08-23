import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import matplotlib.pyplot as plt
import os
class ExtractBoundaries:
    def __init__(self,file_path):
        self.file_path=file_path
        self.img_path='example.jpg'
        self.square_boundry=[]
        self.semi_circle_boundry=[]
        self.circle_boundry=[]
        self.ellipse_boundry=[]
        self.star_boundry=[]
        self.other_shapes_boundry=[]
        self.triangle_boundry=[]
        self.tol=0.1
        self.plot()
        self.get_contours()
        

    def read_csv(self):
        np_path_XYs = np.genfromtxt(self.file_path, delimiter=',', skip_header=1)
        
        path_XYs = []
        
        for i in np.unique(np_path_XYs[:, 0]):
            npXYs = np_path_XYs[np_path_XYs[:, 0] == i][:, 1:]
            
            XYs = []
            
            for j in np.unique(npXYs[:, 0]):
                XY = npXYs[npXYs[:, 0] == j][:, 1:]
                XYs.append(XY)
            
            path_XYs.append(XYs)
        
        return path_XYs
        
        
    def plot(self):
        colours=['black']
        fig, ax = plt.subplots(tight_layout=True, figsize=(8, 8))
        
        for i, XYs in enumerate(self.read_csv()):
            c = colours[i % len(colours)]
            for XY in XYs:
                ax.plot(XY[:, 0], XY[:, 1], c=c, linewidth=2)
        
        ax.set_aspect('equal')
        
        plt.axis("off")
        plt.savefig(self.img_path,format='jpg')
        
        
    def rescale_img(self,frame):
        h,w=frame.shape[:2]
        new_h=int(h*0.5)
        new_w=int(w*0.5)
        return cv.resize(frame,(new_h,new_w))
        
        
    def get_contours(self):
        img=cv.imread(self.img_path)
        self.re_img=self.rescale_img(img)
        blur_img=cv.GaussianBlur(self.re_img,(5,5),2)
        gray_img=cv.cvtColor(blur_img,cv.COLOR_BGR2GRAY)
        canny_img=cv.Canny(gray_img,150,100)
        contours,hierarchy=cv.findContours(canny_img,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
        ind=0
        
        for contour in contours:
            epsilon=0.03*cv.arcLength(contour,True)
            approx=cv.approxPolyDP(contour,epsilon,True)
            
            if len(approx)>=1 and len(approx)<=2:
                pass
            elif len(approx)==3:
                self.triangle_boundry.append(contour)
                
            elif len(approx)==4:
                x,y,w,h=cv.boundingRect(contour)
                area_enclosing=w*h
                peri_enclosing=2*(w+h)
                
                area_contour=cv.contourArea(contour)
                peri_contour=cv.arcLength(contour,True)
                
                area_ratio=area_contour/area_enclosing
                peri_ratio=peri_contour/peri_enclosing
                
                if (1-self.tol)<=area_ratio<=(1+self.tol) and (1-self.tol)<=area_ratio<=(1+self.tol):
                    self.square_boundry.append(contour)
                
            elif len(approx)==5 or len(approx)==6 or len(approx)==7:
                #Check for Ellipse:

                area_contour=cv.contourArea(contour)
                ellipse=cv.fitEllipse(contour)
                (x,y),(major_axis,minor_axis),angle=cv.fitEllipse(contour)
                area_ellipse=np.pi*(major_axis/2)* (minor_axis/2)
                
                area_ratio_ellipse=area_contour/area_ellipse
                
                if (self.tol-1)<=area_ratio_ellipse<=(self.tol+1):
                    self.ellipse_boundry.append(contour)
            
            elif len(approx)>7 and len(approx)<=9:
                area_contour=cv.contourArea(contour)
                peri_contour=cv.arcLength(contour,True)
                
                #Check for circle:''

                (x,y),radius=cv.minEnclosingCircle(contour)
                area_circle=np.pi*(radius**2)
                peri_circle=2*np.pi*radius
                
                area_ratio_circle=area_contour/area_circle
                peri_ratio_circle=peri_contour/peri_circle
                
                #Check for semicirlce:
                area_semi=np.pi*(radius**2)/2
                peri_semi=np.pi*radius
                
                area_ratio_semi=area_contour/area_semi
                peri_ratio_semi=peri_contour/peri_semi
                
                if (self.tol-1)<=area_ratio_semi<=(self.tol+1) and  (self.tol-1)<=peri_ratio_semi<=(self.tol+1):
                    self.semi_circle_boundry.append(contour)
                    
                elif (self.tol-1)<=area_ratio_circle<=(self.tol+1) and  (self.tol-1)<=peri_ratio_circle<=(self.tol+1):
                    self.circle_boundry.append(contour)
                    
               
                
                
                
            elif len(approx)==10:
                self.star_boundry.append(contour)
                
            else:
                self.other_shapes_boundry.append(contour)
                
                
    
            
        
    def get_deserving_contours(self,contours):
        deserving_contours=[]
        contour_info=[]
        for contour in contours:
            M = cv.moments(contour)
            if M["m00"] != 0:
                cX=int(M["m10"] / M["m00"])
                cY=int(M["m01"] / M["m00"])
            else:
                cX,cY=0,0
            centre=(cX,cY)
            area=cv.contourArea(contour)
            contour_info.append((contour,area,centre))
            
        contour_info=sorted(contour_info,key=lambda x:x[1])
        contour_info.reverse()
        
        not_visited_centre=[]
        for cont_info in contour_info:
            if cont_info[2] not in not_visited_centre:
                deserving_contours.append(cont_info[0])
                not_visited_centre.append(cont_info[2])
                
        return deserving_contours
        
        
    def get_circle_boundry(self):
        if len(self.circle_boundry)!=0:
            data=self.get_deserving_contours(self.circle_boundry)
            out=[np.array(cont).squeeze(axis=1) for cont in data]
            print(f"{len(out)} contours are found")
            return out
            
        else:
            return self.circle_boundry
            
    def get_triangle_boundry(self):
        if len(self.triangle_boundry)!=0:
            data=self.get_deserving_contours(self.triangle_boundry)
            out=[np.array(cont).squeeze(axis=1) for cont in data]
            print(f"{len(out)} contours are found")
            return out
        else:
            return self.triangle_boundry
        
    def get_semi_circle_boundry(self):
        if len(self.semi_circle_boundry)!=0:
            data=self.get_deserving_contours(self.semi_circle_boundry)
            out=[np.array(cont).squeeze(axis=1) for cont in data]
            print(f"{len(out)} contours are found")
            return out
        else:
            return self.semi_circle_boundry
            
        
         
    def get_square_boundry(self):
        if len(self.square_boundry)!=0:
            data=self.get_deserving_contours(self.square_boundry)
            out=[np.array(cont).squeeze(axis=1) for cont in data]
            print(f"{len(out)} contours are found")
            return out
        else:
            return self.square_boundry 
        
    def get_ellipse_boundry(self):
        if len(self.ellipse_boundry)!=0:
            data=self.get_deserving_contours(self.ellipse_boundry)
            out=[np.array(cont).squeeze(axis=1) for cont in data]
            print(f"{len(out)} contours are found")
            return out
        else:
            return self.ellipse_boundry
        
    def get_star_boundry(self):
        if len(self.star_boundry)!=0:
            data=self.get_deserving_contours(self.star_boundry)
            out=[np.array(cont).squeeze(axis=1) for cont in data]
            print(f"{len(out)} contours are found")
            return out
        else:
            return self.star_boundry  
        
    def get_other_shapes_boundry(self):
        if len(self.other_shapes_boundry)!=0:
            data=self.get_deserving_contours(self.other_shapes_boundry)
            out=[np.array(cont).squeeze(axis=1) for cont in data]
            print(f"{len(out)} contours are found")
            return out
        else:
            return self.other_shapes_boundry

    
        
    def plot_boundry(self,contour):

        if len(contour) != 0:
            num_contours = len(contour)
            
           
            num_cols = 3  
            num_rows = (num_contours + num_cols - 1) // num_cols 
        
            fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5*num_rows))
            axes = axes.flatten()  
        
            for idx, cont in enumerate(contour):
                cont = np.expand_dims(cont, axis=1)
                blank_img = np.zeros_like(self.re_img)
                cv.drawContours(blank_img, [cont], -1, (0, 255, 0), 2)
                
               
                blank_img_rgb = cv.cvtColor(blank_img, cv.COLOR_BGR2RGB)
                
                
                axes[idx].imshow(blank_img_rgb)
                axes[idx].set_title(f'Contour {idx+1}')
                axes[idx].axis('off')  
        
           
            for ax in axes[num_contours:]:
                ax.axis('off')
        
            plt.tight_layout()
            plt.show()
        else:
            print('No contours available')
            
    def return_rescale_img(self):
        return self.re_img
    
            


    

                


                

                
            
        
        
    
