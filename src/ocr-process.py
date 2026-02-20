# Based upon project work by Claire Su and William Qiu at https://www.idapgh.org/data-analysis-of-the-allegheny-observatory
# Processes the embedded text in the image data into a dataframe and saves it to a CSV file
import os
from PIL import Image
from PIL.Image import Resampling
from pytesseract import pytesseract
from pandas import DataFrame

if __name__ == '__main__':
    # Finds the name of all images in a folder
    filePath = r'C:\Users\Rainybyte\Research\AllSky Light Pollution Project\2026-02-20\\'
    allImages = os.listdir(filePath)

    # Important data lists
    date = []
    time = []
    exposure = []
    imageRGB = []
    imageNumber = []

    tesseractPath = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

    images = []
    for file in allImages:
        imagePath = filePath + file
        print(imagePath)
        image = Image.open(imagePath)
        images.append((file, image))

    for (file, image) in images:
        print(file)

        full_height = 480
        block_width = 80
        top_height = 54
        bottom_height = 20

        top_block = image.crop((0, 0, block_width, top_height))
        top_block = top_block.convert('L')
        bottom_block = image.crop((0, full_height - bottom_height, block_width, full_height))
        bottom_block = bottom_block.convert('L')

        stat_block = Image.new('RGB', (block_width, top_height + bottom_height))
        stat_block.paste(top_block, (0, 0))
        stat_block.paste(bottom_block, (0, top_height))
        stat_block = stat_block.resize((2 * block_width, 2 * (top_height + bottom_height)), Resampling.LANCZOS)

        # Directs tesseract to the library
        pytesseract.tesseract_cmd = tesseractPath

        # Extracts the text
        newData = pytesseract.image_to_string(stat_block)
        newData = newData.splitlines()

        # Between here and the pixel brightness computations, stuff gets unexplainably wonky. Use dataframe


        data = []
        for i in range(len(newData) - 1):
            newData[i] = newData[i].replace(' ', '')
            iterData = newData[i].replace(':', '')
            if len(iterData) >= 4:
                data.append(iterData)

        # Makes sure the list is not empty
        emptyCounter = 0
        while len(data) <= 2:
            data.append('0')

        # Removes the file number
        if len(data[-1]) == 9:
            data.remove(data[-1])

        # Adds each date
        try:
            if len(data[0]) >= 8:
                date.append(data[0])
            else:
                date.append('Check ' + file)
        except:
            date.append('Check ' + file)

        # Adds each time
        try:
            if data[1].isdecimal() == True:
                time.append(int(data[1]))
            else:
                time.append('Check ' + file)
        except:
            time.append('Check ' + file)

        # Adds each exposure
        try:
            expoCounter = 0
            for i in data[2:]:
                if len(i) >= 5 and len(i) <= 7:
                    correctedExposure = (float(i) - 5) / 100000
                    exposure.append(correctedExposure)
                    expoCounter += 1
            if expoCounter == 0:
                exposure.append('Check ' + file)
        except:
            exposure.append('Check ' + file)

        # Averages of the pixels in RGB
        pixelNW = image.getpixel((310, 230))
        pixelNE = image.getpixel((330, 230))
        pixelSW = image.getpixel((310, 250))
        pixelSE = image.getpixel((330, 250))
        pixelCenter = image.getpixel((320, 240))
        pixelAvg = []
        for i in range(3):
            pixelAvg.append((pixelNW[i] + pixelNE[i] + pixelSW[i] + pixelSE[i] + pixelCenter[i]) / 5)

        # Adds each pixel average
        imageRGB.append(pixelAvg)

        # Adds each file name
        imageNumber.append(file)

    # Converts the following lists to a CSV
    df = DataFrame({
        "Date": date,
        "Time": time,
        "Exposure Time (s)": exposure,
        "Average Pixel Brightness": imageRGB,
        "File Number": imageNumber
    })
    df.to_csv('allsky.csv')

    print('done')

