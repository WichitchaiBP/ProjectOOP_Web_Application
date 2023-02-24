import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import joblib
import io
import os
import torch
import wget
from sklearn.linear_model import LinearRegression
from PIL import Image
from torchvision import transforms
import random

options = ["HOME","The Cost of Building","Machine Learning"]

selected_option = st.sidebar.selectbox("Menu", options)
if selected_option == "HOME":
    P1 = st.sidebar
    if P1:
        st.markdown("<h2 style='color: gray;'>วิธีคำนวณราคาสร้างบ้าน “อย่างง่าย” งบเท่าไหร่ถึงจะพอ\n</h2>",
                    unsafe_allow_html=True)
        st.markdown("<h4 style='color: #0E2A1A  ;'>บ้านในฝันของเรา ต้องลงทุนเท่าไหร่กัน ?</h4>",
                    unsafe_allow_html=True)
        st.markdown("<h6 style='color: #0E2A1A   ;'>เป็นคำถามที่พบกันบ่อยครั้ง แบบบ้านที่นำมาโชว์บนเว็บไซต์บ้านไอเดีย ราคาเท่าไหร่ อันที่จริงทางเว็บเองต้องเรียนตามตรงว่า ไม่อาจทราบราคาจริงได้อย่างชัดเจน หากจะตอบก็ตอบได้เพียงแค่ราคาประมาณเบื้องต้นเท่านั้น เพราะหากต้องการราคาจริง จำเป็นต้องรู้รายละเอียดของวัสดุ ระบบโครงสร้าง ปัญหาหน้างาน และอีกหลาย ๆ ปัจจัยที่ไม่อาจทราบได้ จึงอยากให้ผู้ชมได้ชมกันไว้เป็นแนวทางในการต่อยอด เก็บเกี่ยวไอเดียต่าง ๆ ไว้ประยุกต์ใช้งานเพื่อบ้านในฝันของเราเองย่อมดีกว่า สำหรับเนื้อหาชุดนี้ ขอแนะนำวิธีประมาณราคาบ้านในฝันเบื้องต้นของเรากัน โดยจะเหมาะกับผู้ที่เป็นเจ้าของบ้าน ซึ่งอาจจะไม่ได้รู้รายละเอียดเบื้องลึกของการสร้างบ้านมากนัก แต่ก็พอจะสามารถตีเป็นตัวเลขกลม ๆ ได้เช่นกัน\n</h6>",
                    unsafe_allow_html=True)
        st.markdown("<h6 style='color: #0E2A1A ;'>ก่อนจะคำนวณราคาบ้าน สิ่งแรกที่เราควรต้องรู้ คือ รู้ความต้องการของเราเอง บ้านในฝันของเรา มีสมาชิกทั้งหมดกี่คน ต้องการห้องนอนกี่ห้อง ห้องน้ำกี่ห้อง ห้องน้ำภายในห้องนอนด้วยมั้ย เป็นคนชอบทำอาหารมากน้อยแค่ไหน กิจกรรมใดที่ต้องมีในชีวิตประจำวัน เราจำเป็นต้องมาแจกแจงรายละเอียดต่าง ๆ ในความต้องการของเราเอง ซึ่งเป็นสิ่งแรกที่เราควรทราบก่อนที่จะหาแบบบ้านที่ชอบ เพราะแบบบ้านที่ชอบ ฟังก์ชั่นการใช้งานอาจไม่ได้ตรงกับที่เราต้องการใช้จริง\n</h4>",
                    unsafe_allow_html=True)
        st.image("https://wp-assets.dotproperty.co.th/wp-content/uploads/sites/9/2016/05/27035531/01_resize6.jpg")
        st.markdown("<h4 style='color: black ;'>ตัวอย่าง\n</h4>",
                    unsafe_allow_html=True)
        st.markdown('<h6 style="color: black ;">“คุณบี ทำงานเป็นนักเขียนอิสระ ต้องอาศัยอยู่ภายในบ้านตลอดทั้งวัน ทั้งการใช้ชีวิตและการทำงาน ปัจจุบันมีลูกสาว 1 คน และมีแผนไว้ว่าจะมีเพิ่มอีก 1 คนใน 2 ปีข้างหน้า ส่วนสามีทำงานบริษัท เป็นพนักงานประจำ ทุก ๆ เย็นจะต้องกลับมาลิ้มรสฝีมือการทำอาหารของภรรยา”\n</h6>',
                    unsafe_allow_html=True)
        st.markdown("<h6 style='color: black ;'>ด้วยลักษณะการอยู่อาศัยของคุณบี ต้องอยู่บ้านตลอดทั้งวัน บ้านจึงเป็นส่วนสำคัญของชีวิต และลักษณะงานของนักเขียน สิ่งที่จำเป็นต้องมือคือมุมสงบไว้สำหรับนั่งอ่านหนังสือ นั่งเขียนงาน มุมดังกล่าวควรมีลักษณะเปิดโปร่งเพื่อให้เกิดความคิดสร้างสรรค์และจินตนาการที่กว้างไกล พอจะสรุปความต้องการเบื้องต้นได้ ดังนี้\n</h6>",
                    unsafe_allow_html=True)
        st.markdown("<h6 style='color: black ;'>ปัจจุบันอยู่ร่วมกัน 3 คน พ่อแม่ลูก ห้องนอนที่จำเป็นตอนนี้มี 2 ห้อง แต่แผนในอนาคตต้องการไว้อีก 1 คน จึงควรออกแบบเผื่อไว้เป็น 3 ห้อง โดยห้องพ่อแม่ เลือกเป็น Master Bedroom ขนาดความกว้างประมาณ 4×6 เมตร มีห้องน้ำในตัว ส่วนห้องลูก ๆ ขนาด 3×4 เมตร ห้องนอนของลูกทั้งสอง ใช้ห้องน้ำร่วมกัน ห้องน้ำมีขนาด 2×2 เมตร\n</h6>",
                    unsafe_allow_html=True)
        st.markdown("<h6 style='color: black ;'>เนื่องด้วยต้องทำอาหารทานเองทุกวัน ห้องครัวจึงเป็นห้องที่ต้องให้ความสำคัญเป็นอย่างยิ่ง ควรออกแบบให้มีความกว้างเพื่อให้เกิดการใช้งานสะดวก โดยคุณบีเลือกที่จะให้ครัวมีความกว้างยาวประมาณ 4×3 เมตร ครัวเชื่อมต่อกับห้องนั่งเล่นที่มีความกว้างประมาณ 3×3 เมตร พร้อมกับห้องน้ำรวมซึ่งอยู่ใกล้ ๆ กัน\n</h6>",
                    unsafe_allow_html=True)
        st.markdown("<h6 style='color: black ;'>ห้องทำงานเขียน เป็นห้องที่คุณบีต้องหมกตัวอยู่ในห้องดังกล่าวเกือบทั้งวัน ห้องนี้จึงเลือกวางไว้ในตำแหน่งทิศเหนือ ซึ่งจะช่วยหลบแสงแดดยามบ่ายได้ดี ตัวห้องเชื่อมต่อกับระเบียงสวนข้างบ้านและผังของห้องเชื่อมต่อกับห้องนอนใหญ่ แต่การใช้งานส่วนใหญ่จะนั่งในห้องนี้เพียงคนเดียว จึงออกแบบไว้ขนาดเล็กเพียง 3×3 เมตรเท่านั้น\n</h6>",
                    unsafe_allow_html=True)
        st.markdown("<h6 style='color: black ;'>ที่บ้านมีรถยนต์ 1 คัน จักรยานยนต์ 1 คัน สามีเลือกที่จะให้โรงจอดรถออกแบบเป็นส่วนเดียวกันกับตัวบ้าน โดยขนาดของโรงจอดรถ มีความกว้าง 5.5 x 6 เมตร เผื่อพื้นที่ไว้สำหรับจักรยานยนต์\n</h6>",
                    unsafe_allow_html=True)
        st.image("https://wp-assets.dotproperty.co.th/wp-content/uploads/sites/9/2016/05/27035533/02_resize6.jpg")
        st.markdown("<h6 style='color: black ;'>ทั้งหมดนี้คือโจทย์ของความต้องการในการใช้งานจริง หากเรานำมาพิจารณาถึงพื้นที่ใช้สอยของแต่ละห้อง เราก็จะได้พื้นที่ใช้สอยรวมโดยประมาณ โดยบ้านของคุณบีเมื่อนำมาคำนวณแล้ว จะมีพื้นที่ใช้สอยทั้งหมดประมาณ 113 ตารางเมตร เผื่อพื้นที่ระเบียงและอื่น ๆ รวมไว้ประมาณ 125 ตารางเมตร\n</h6>",
                    unsafe_allow_html=True)
        st.markdown("<h6 style='color: black ;'>เมื่อได้ขนาดของพื้นที่ใช้สอยแล้ว ให้นำมาคูณกับ ค่าใช้จ่ายในการก่อสร้างแบบประมาณการ โดยปกติค่าก่อสร้างมักคิดตามสเปคของวัสดุ และดีไซน์ของบ้าน ดังตัวอย่างต่อไปนี้\n</h6>",
                    unsafe_allow_html=True)
        st.markdown("<h6 style='color: black ;'>บ้านวัสดุแบบทั่วไป ค่าแรง+ค่าวัสดุ = 9,000 – 12,000 บาท / ตร.ม.\n</h6>",
                    unsafe_allow_html=True)
        st.markdown("<h6 style='color: black ;'>บ้านวัสดุระดับกลาง  ค่าแรง+ค่าวัสดุ = 13,000 – 15,000 บาท / ตร.ม.\n</h6>",
                    unsafe_allow_html=True)
        st.markdown("<h6 style='color: black ;'>บ้านวัสดุระดับพรีเมี่ยม ค่าแรง+ค่าวัสดุ = 15,000 – 20,000 บาท / ตร.ม.\n</h6>",
                    unsafe_allow_html=True)
        st.markdown("<h6 style='color: black ;'>* เป็นเพียงเกณฑ์ราคาทั่วไป การสร้างจริงยังมีราคาที่สูงกว่านี้ โดยเฉพาะบ้านที่เน้นงานสถาปัตยกรรม\n</h6>",
                    unsafe_allow_html=True)
        st.markdown("<h6 style='color: black ;'>ผู้อ่านอาจสงสัยอีกว่า แล้ววัสดุทั่วไป วัสดุระดับกลาง วัสดุพรีเมี่ยมนั้นเป็นแบบไหน จะว่าไปแต่ละบริษัทก็มีมาตรฐานสเปคที่แตกต่างกันออกไป จึงขอยกตัวอย่างวัสดุบางส่วน เพื่อให้ผู้อ่านนึกภาพออกมากยิ่งขึ้น โดยในตัวอย่างนี้ เป็นสเปคมาตรฐานของแบรนด์ Focus House หากผู้อ่านเลือกใช้บริการของบริษัทไหน ต้องตรวจเช็คกับบริษัทนั้น ๆ อีกทีนะครับ\n</h6>",
                    unsafe_allow_html=True)
        st.text("\n")
        st.markdown("<h6 style='color: black ;'>วัสดุปูพื้น\n</h6>",
                    unsafe_allow_html=True)
        st.markdown("<h6 style='color: black ;'>บ้านทั่วไป ปูพื้นกระเบื้อง ระดับกลาง ปูแกรนิโต้ ระดับพรีเมี่ยม พื้นลามิเนต หรืออื่น ๆ ที่สูงกว่า\n</h6>",
                    unsafe_allow_html=True)
        st.text("\n")
        st.markdown("<h6 style='color: black ;'>วัสดุหลังคา\n</h6>",
                    unsafe_allow_html=True)
        st.markdown("<h6 style='color: black ;'>บ้านทั่วไป หลังคาโมเนีย ระดับกลาง เพรสทีท ระดับพรีเมี่ยม นิวสไตล์ครับ (CPAC roof) ส่วนเมทัลชีทจะถูกกว่าวัสดุอื่น ๆ\n</h6>",
                    unsafe_allow_html=True)
        st.text("\n")
        st.markdown("<h6 style='color: black ;'>สีทาบ้าน\n</h6>",
                    unsafe_allow_html=True)
        st.markdown("<h6 style='color: black ;'>บ้านทั่วไป TOA 4 Season ระดับกลาง TOA Shield 1 ระดับพรีเมียม  TOA Super shield\n</h6>",
                    unsafe_allow_html=True)
        st.text("\n")
        st.markdown("<h6 style='color: black ;'>ข้อมูลข้างต้นเป็นเพียงตัวอย่างของสเปคบ้าน ซึ่งแต่ละบริษัทอาจเลือกแบรนด์วัสดุที่แตกต่างกัน แต่ละแบรนด์ก็จะมีหลายเกรดให้เราเลือก กรณีที่เจ้าของบ้านต้องการปรับเปลี่ยนวัสดุบางส่วนสามารถแจ้งกับผู้รับเหมาได้ หากเกรดสูงกว่าก็จะเพิ่มเงินเฉพาะส่วน หรือหากเกรดต่ำกว่า ก็จะลดราคาให้ แต่ทั้งนี้ก็ต้องขอย้ำกันอีกครั้งว่า เป็นเพียงตัวอย่างเท่านั้น เนื่องด้วยแต่ละบริษัทรับสร้างบ้าน จะมีเกณฑ์มาตรฐานที่แตกต่างกัน ขึ้นอยู่กับประสบการณ์ งานฝีมือ ระบบ และหลาย ๆ ปัจจัย\n</h6>",
                    unsafe_allow_html=True)
        st.markdown("<h6 style='color: black ;'>นอกจากวัสดุแล้ว ยังต้องดูดีไซน์ของบ้านเป็นส่วนประกอบ หากงานสถาปัตยกรรมมีความละเอียดอ่อน มีดีไซน์ที่เป็นผลให้การดำเนินก่อสร้างยากขึ้น หรือใช้เทคโนโลยีอื่น ๆ มาเกี่ยวข้อง เช่น บ้านโครงสร้างเหล็ก, บ้านผนังหล่อคอนกรีต หรือเทคโนโลยีอื่น ๆ ที่สูงกว่า ค่าก่อสร้างก็จะถูกปรับสูงขึ้นเช่นกัน\n</h6>",
                    unsafe_allow_html=True)
        st.text("\n")
        st.markdown("<h6 style='color: black ;'>เมื่อทราบถึงเกรดวัสดุของบ้านแล้ว ให้ผู้อ่านนำพื้นที่ใช้สอยไปคูณกับค่าสร้างบ้าน+ค่าวัสดุ เช่น บ้านคุณบี ไม่เน้นหรูมากนัก แต่ก็อยากได้บางส่วนที่เป็นเอกลักษณ์อยู่บ้าง เลือกสร้างบ้านระดับกลาง สมมุติ ค่าแรง + ค่าวัสดุ =  14,000 บาท/ ตร.ม.\n</h6>",
                    unsafe_allow_html=True)
        st.markdown("<h6 style='color: black ;'>ค่าสร้างบ้านจะได้ = 125 ตร.ม. x 14,000 บาท\n</h6>",
                    unsafe_allow_html=True)
        st.markdown("<h6 style='color: black ;'>คุณบีจะต้องมีเงินจำนวน 1.75 ล้านบาท เพื่อใช้ในการสร้างบ้านในฝัน 1 หลัง\n</h6>",
                    unsafe_allow_html=True)
        st.image("https://wp-assets.dotproperty.co.th/wp-content/uploads/sites/9/2016/05/27035536/03_resize5.jpg")
        st.markdown("<h6 style='color: black ;'>ทั้งหมดนี้เป็นค่าใช้จ่ายในการก่อสร้าง ยังต้องมีค่าออกแบบกรณีจ้างสถาปนิก,วิศวกร, ค่าใช้จ่ายในการตกแต่งภายใน การตกแต่งภายใน ซื้อเฟอร์นิเจอร์ เครื่องใช้ไฟฟ้าและของใช้อื่น ๆ เพราะฉะนั้นผู้สร้างบ้านจึงต้องเตรียมเงินไว้สำหรับการตกแต่งภายในต่างหาก และเตรียมเงินไว้สำหรับกรณีการก่อสร้างจริงที่อาจเป็นไปได้ว่างบจะบาน เพราะบ่อยครั้งเมื่อดำเนินการสร้างไป เจ้าของบ้านได้ไอเดียเพิ่ม อยากเพิ่มโน่น เพิ่มนี่ หรืออาจติดปัญหาในกระบวนการก่อสร้าง จึงจำเป็นต้องเผื่อเงินก้อนนี้ไว้ด้วยครับ\n</h6>",
                    unsafe_allow_html=True)
        st.text("\n")
        st.markdown("<h6 style='color: black ;'>โดยใน Menu ผู้อ่านสามารถใช้โปรแกรมการคำนวณง่ายๆ เพื่อหาราคาคร่าวๆ ในการสร้างบ้าน 1 หลัง\n"
                    "และทางเว็บไซต์ของเรายังมี Matchine Learning ให้ผู้อ่านสามารถเข้าไป Trian ได้อีกด้วย...</h6>",
                    unsafe_allow_html=True)

        st.markdown(
            f"""
                   <style>
                   .stApp {{
                       background-image: url("https://images.pexels.com/photos/323645/pexels-photo-323645.jpeg?blur=200");
                       background-attachment: fixed;
                       background-size: cover;
                       /* opacity: 0.3; */
                   }}
                   </style>
                   """,
            unsafe_allow_html=True)

if selected_option == "The Cost of Building":
    P2 = st.sidebar
    if P2:
        st.markdown(
            f"""
               <style>
               .stApp {{
                   background-image: url("https://images.pexels.com/photos/257643/pexels-photo-257643.jpeg?blur=50");
                   background-attachment: fixed;
                   background-size: cover;
                   /* opacity: 0.3; */
               }}
               </style>
               """,
            unsafe_allow_html=True
        )
        st.markdown(
            """
            <h1 style='text-align: center'>The Cost of Building a House Prediction</h1>
            """,
            unsafe_allow_html=True
        )

        st.text("เว็บนี้จะช่วยบอกราคาคร่าวๆ ในการสร้างบ้าน 1 หลัง ")
        st.text("โดยที่ผู้ใช้กรอกราคาวัสดุ ค่าช่าง (ต่อตารางเมตร) ")

        matirial = st.number_input("ค่าวัสดุ(ต่อตารางเมตร):", )
        wage = st.number_input("ค่าช่าง(ต่อตารางเมตร):")
        area = st.number_input("พื้นที่(ตารางเมตร):")
        x2 = st.button("Calculate")
        if x2:
            calculator = st.write(f'ราคาในการสร้างบ้าน 1 หลัง พื้นที่ {area} ตารางเมตร ต้องใช้เงินทั้งหมด {(matirial+wage)*area:.2f} บาท')

        def load_house_data():
            return pd.read_excel('House.xlsx')

        def save_model(model):
            joblib.dump(model, 'model.joblib')

        def load_model():
            return joblib.load('model.joblib')

        def generate_house_data():
            rng = np.random.RandomState(0)
            n = 10
            n1 = random.randrange(200, 400) # ค่าของ
            n2 = random.randrange(300, 400) # ค่าช่าง
            n3 = random.randrange(400, 2000) # พื้นที่
            n4 = (n1 + n2) * n3 #ค่าใช้จ่าย
            x = np.round(n3 * rng.rand(n))
            y = np.round(n4 * rng.rand(n))  # พื้นที่
            df = pd.DataFrame({
                'Expenses': y,
                'Area': x
            })
            df.to_excel('House.xlsx')

        generateb = st.button('generate House.xlsx')
        if generateb:
            st.write('generating "House.xlsx" ...')
            generate_house_data()

        loadb = st.button('load House.xlsx')
        if loadb:
            st.write('loading "House.xlsx ..."')
            df = pd.read_excel('House.xlsx', index_col=0)
            st.dataframe(df)
            fig, ax = plt.subplots()
            df.plot.scatter(x='Area', y='Expenses', ax=ax)
            st.pyplot(fig)

        trainb = st.button('Train Program')
        if trainb:
            st.write('Training Model ...')
            df = pd.read_excel('House.xlsx', index_col=0)
            model = LinearRegression()
            st.dataframe(df)
            save_model(model)

        chart = st.button('Chart Program')
        if chart:
            tab1, tab2, tab3 = st.tabs(["Line Chart", "Area Chart", "Bar Chart"])
            df = pd.read_excel('House.xlsx', index_col=0)
            tab1.line_chart(df)
            tab2.area_chart(df)
            tab3.bar_chart(df)


if selected_option == "Machine Learning":
    P3 = st.sidebar
    if P3:
        st.markdown(
            f"""
               <style>
               .stApp {{
                   background-image: url("https://images.pexels.com/photos/9555137/pexels-photo-9555137.jpeg?blur=50");
                   background-attachment: fixed;
                   background-size: cover;
                   /* opacity: 0.3; */
               }}
               </style>
               """,
            unsafe_allow_html=True
        )


        def load_image():
            uploaded_file = st.file_uploader(label='Pick an image to test')
            if uploaded_file is not None:
                image_data = uploaded_file.getvalue()
                st.image(image_data)
                return Image.open(io.BytesIO(image_data))
            else:
                return None


        def load_model():
            model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
            model.eval()
            return model


        def load_labels():
            labels_path = 'https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt'
            labels_file = os.path.basename(labels_path)
            if not os.path.exists(labels_file):
                wget.download(labels_path)
            with open(labels_file, "r") as f:
                categories = [s.strip() for s in f.readlines()]
                return categories


        def predict(model, categories, image):
            preprocess = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            input_tensor = preprocess(image)
            input_batch = input_tensor.unsqueeze(0)

            with torch.no_grad():
                output = model(input_batch)

            probabilities = torch.nn.functional.softmax(output[0], dim=0)

            top5_prob, top5_catid = torch.topk(probabilities, 5)
            for i in range(top5_prob.size(0)):
                st.write(categories[top5_catid[i]], top5_prob[i].item())


        def main():
            st.title('Pretrained model demo')
            model = load_model()
            categories = load_labels()
            image = load_image()
            result = st.button('Run on image')
            if result:
                st.write('Calculating results...')
                predict(model, categories, image)
        main()

