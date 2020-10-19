# -*- coding: utf-8 -*-
"""
Created on Thu May 02 12:39:17 2019

@author: Dell
"""

import sys, os 
import pandas as pd

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "book.settings")

import django
django.setup()

from reviews.models import Book 


def save_book_from_row(book_row):
    book = Book()
    book.id = book_row[0]
    book.name = book_row[1]
    book.save()
    
    
if __name__ == "__main__":
    
    if len(sys.argv) == 2:
        print "Reading from file " + str(sys.argv[1])
        book_df = pd.read_csv(sys.argv[1])
        print book_df

        book_df.apply(
            save_book_from_row,
            axis=1
        )

        print "There are {} book".format(Book.objects.count())
        
    else:
        print "Please, provide Book file path"