'''
=================================================================================================================================================================


                                                            BOOK RECOMMENDER SYSTEM - SVD and KNN

                                                                                                                                            Aayushi Gupta
                                                                                                                                            Girish Chhabra
                                                                                                                                            Vineet Khatwal
=================================================================================================================================================================

'''
import Model1_SVDnKNN
import argparse
import sys
import pandas as pd
import os
import pandas as pd
import numpy as np
import stdiomask
import smtplib
import ssl
from scipy.sparse.linalg import svds
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
import sys
from pyspark import SparkConf, SparkContext
from math import sqrt
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os
os.environ["PYSPARK_PYTHON"] = "/usr/local/bin/python3"


def parse_arguments():

    print("********************        Parsing the Arguments         ***************************")
    argumentParser = argparse.ArgumentParser(
        description='SVD or KNN collaborative filtering')

    argumentParser.add_argument(
        "--SVD",                                 # "If argument if SVD"
        action="store_true",
        help="User based collaborative filtering using SVD"
    )

    argumentParser.add_argument(
        "--KNN",                                # "If argument if KNN"
        action="store_true",
        help="Item collaborative filtering using KNN"
    )

    return argumentParser.parse_args()


def YN():

    continueExporingRecommendations = str(
        input('\n\nContinue (y/n):\t')).lower().strip()
    if continueExporingRecommendations[0] == 'y':
        return True
    if continueExporingRecommendations[0] == 'n':
        return False
    else:
        return False


def sendMail(recommendationList):
    sender_email = "cmpe.256.recommendation@gmail.com"
    #receiver_email = "vineet.khatwal@sjsu.edu", "aayushi.gupta@sjsu.edu", "girish.chhabra@sjsu.edu"
    receiver_email = "vineet.khatwal@sjsu.edu"
    print("Type your password and press enter:")
    password = stdiomask.getpass(mask='X')
    message = MIMEMultipart("alternative")
    message["Subject"] = "Hi Reader ! Recommendation for you from CMPE 256 : Team 5"
    message["From"] = "cmpe.256.recommendation@gmail.com"
    message["To"] = ", ".join(receiver_email)

    print("================== Composing the mail ================== ")
    # Create the plain-text and HTML version of your message
    html = """\
    <html>
        <head></head>
        <body>
            {0}
        </body>
    </html>
    """.format(recommendationList.to_html())

    # Turn these into plain/html MIMEText objects
    part1 = MIMEText(html, 'html')

    # Add HTML/plain-text parts to MIMEMultipart message
    # The email client will try to render the last part first
    message.attach(part1)

    # Create secure connection with server and send email
    context = ssl.create_default_context()
    with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
        server.login(sender_email, password)
        server.sendmail(
            sender_email, receiver_email, message.as_string()
        )

    print("==================   Main sent ================== ")


def sendMail(recommendationList):
    sender_email = "cmpe.256.recommendation@gmail.com"
    receiver_email = "vineet.khatwal@sjsu.edu", "aayushi.gupta@sjsu.edu", "girish.chhabra@sjsu.edu"
    #receiver_email = "vineet.khatwal@sjsu.edu"
    print("Type your password and press enter:")
    password = stdiomask.getpass(mask='X')
    message = MIMEMultipart("alternative")
    message["Subject"] = "Hi Reader ! Recommendation for you from CMPE 256 : Team 5"
    message["From"] = "cmpe.256.recommendation@gmail.com"
    message["To"] = ", ".join(receiver_email)

    print("================== Composing the mail ================== ")
    # Create the plain-text and HTML version of your message
    html = """\
    <html>
        <head></head>
        <body>
            {0}
        </body>
    </html>
    """.format(recommendationList.to_html())

    # Turn these into plain/html MIMEText objects
    part1 = MIMEText(html, 'html')

    # Add HTML/plain-text parts to MIMEMultipart message
    # The email client will try to render the last part first
    message.attach(part1)

    # Create secure connection with server and send email
    context = ssl.create_default_context()
    with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
        server.login(sender_email, password)
        server.sendmail(
            sender_email, receiver_email, message.as_string()
        )

    print("==================   Main sent ================== ")


def main():
    #print("In Main")

    args = parse_arguments()
    print(" --------------------- Arguments ---------------------")
    print(args)
    cont = True

    if not args.SVD and not args.KNN:
        print("****************************************************************************************************")
        print("                                                                                                    ")
        print("                 For User/Item based collaborative filtering use : SVD or KNN                       ")
        print("                                                                                                    ")
        print("****************************************************************************************************")

        #Top_B = Books()
        TopBooks = Model1_SVDnKNN.AmazonBooks()

        HighMeanRating, HighRatingCount = TopBooks.Top_Books()

        pd.set_option('display.max_colwidth', -1)

        print("****************************************************************************************************")
        print("\n\nHighly Rated Books:\n")
        print(HighMeanRating[['product_title', 'MeanRating', 'ratingCount']])

        sendMail(
            HighMeanRating[['product_title', 'MeanRating', 'ratingCount']])

        print("\n\nHigh Rating count books :\n")
        print(
            HighRatingCount[['product_title', 'MeanRating', 'ratingCount']])
        sendMail(
            HighRatingCount[['product_title', 'MeanRating', 'ratingCount']])

        sys.exit()

    if args.SVD:

        #UCF = SVD()
        SVDModel = Model1_SVDnKNN.SVD()
        # print(UCF)
        SVDModel.SVDBreakDown()

        while cont:
            try:
                Cust_ID = int(
                    input('Enter Customer ID in the range {0}-{1}: '.format(1, len(SVDModel.explicit_users))))
            except:
                print('Enter a number')
                sys.exit()

            if Cust_ID in range(1, len(SVDModel.explicit_users)):
                pass
            else:
                print("Choose between {0}-{1}".format(1,
                                                      len(SVDModel.explicit_users)))
                sys.exit()

            userIDEntered, BookRatedByUser, SVDRecommendedBooks = SVDModel.Recommend_Books(
                userID=Cust_ID)

            pd.set_option('display.max_colwidth', -1)

            '''
            print("Rated_Books")
            print(type(Rated_Books))
            for col in Rated_Books.columns:
                print(col)
            '''

            print("\n" * 3)
            print("*" * 200)
            print("Books which the giver user",
                  userIDEntered, "has already rated")
            print("*" * 200)
            BookRatedByUser = BookRatedByUser.drop_duplicates(
                ['product_id', 'product_title_x'])
            print(BookRatedByUser[['product_title_x',
                                   'product_title_y', 'star_rating_y']])
            print("")
            print("*" * 200)
            print("\n" * 3)
            print("*" * 200)
            print("Recommendation for the user :", userIDEntered)
            print("*" * 200)

            SVDRecommendedBooks = SVDRecommendedBooks.merge(
                SVDModel.average_rating, how='left', on='product_id')
            SVDRecommendedBooks = SVDRecommendedBooks.rename(
                columns={'star_rating': 'Rating'})
            SVDRecommendedBooks = SVDRecommendedBooks.drop_duplicates(
                ['product_id', 'product_title'])
            print(SVDRecommendedBooks[['product_title', 'MeanRating']])
            print("*" * 200)
            print(type(SVDRecommendedBooks))
            sendMail(SVDRecommendedBooks)

            cont = YN()

    if args.KNN:

        #KNNModel = KNN()
        KNNModel = Model1_SVDnKNN.KNN()

        while cont:

            print("*" * 200)
            book_name = input('\nEnter the Book Title:')
            print("*" * 200)

            _, KNN_Recommended_Books, _ = KNNModel.Recommend_Books(book_name)

            print("*" * 200)
            print('Recommendations for the book --> {0}:\n'.format(book_name))
            print("*" * 200)
            KNN_Recommended_Books = KNN_Recommended_Books.merge(
                KNNModel.average_rating, how='left', on='product_id')
            KNN_Recommended_Books = KNN_Recommended_Books.rename(
                columns={'bookRating': 'MeanRating'})

            print("*" * 200)
            print(KNN_Recommended_Books[['product_title', 'MeanRating']])
            print("*" * 200)
            sendMail(KNN_Recommended_Books)
            cont = YN()


if __name__ == '__main__':
    main()
