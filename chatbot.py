#!/usr/bin/env python
# -*- coding: utf-8 -*-

# PA6, CS124, Stanford, Winter 2018
# v.1.0.2
# Original Python code by Ignacio Cases (@cases)
######################################################################
import csv
import math
import re
import sys
import getopt
import os
import operator

import numpy as np

from movielens import ratings
from random import randint
from PorterStemmer import PorterStemmer

class Chatbot:
    """Simple class to implement the chatbot for PA 6."""

    #############################################################################
    # `moviebot` is the default chatbot. Change it to your chatbot's name       #
    #############################################################################
    def __init__(self, is_turbo=False):
      self.name = 'Moviebot'
      self.is_turbo = is_turbo
      self.p = PorterStemmer()
      self.punctuation = set([",", ".","?","!",":",'"',"'","(",")"])
      self.endPunctuation = set([".","?","!",":",'"',"'","(",")"])
      self.negateWords = set(["not", "isn't", "ain't", "aren't", "didn't", "doesn't", "can't", "won't", "no", "never", "neither", "nor", "wasn't"])
      self.prevNegateWords = set(["but", "although", "however", "yet"])
      self.extremeWords = set(["very", "really", "extremely"])
      self.movies = {}
      self.movie_to_index_dict = {}
      self.read_data()

    #############################################################################
    # 1. WARM UP REPL
    #############################################################################

    def greeting(self):
      """chatbot greeting message"""
      #############################################################################
      # TODO: Write a short greeting message                                      #
      #############################################################################

      greeting_message = "Hi! I'm "+self.bot_name()+"! I'm going to recommend a movie to you. First I will ask you about your taste in movies. Tell me about a movie that you have seen."

      #############################################################################
      #                             END OF YOUR CODE                              #
      #############################################################################

      return greeting_message

    def goodbye(self):
      """chatbot goodbye message"""
      #############################################################################
      # TODO: Write a short farewell message                                      #
      #############################################################################

      goodbye_message = 'Have a nice day!'

      #############################################################################
      #                             END OF YOUR CODE                              #
      #############################################################################

      return goodbye_message


    #############################################################################
    # 2. Modules 2 and 3: extraction and transformation                         #
    #############################################################################

    def extract_movie(self, input):
      movie_patt = '\"(.*?)\"'
      matches = re.findall(movie_patt, input)
      movie_name = ""
      response = ""
      processed_sentence = input
      if len(matches) == 0:
        response = "Sorry, I don't understand. Tell me about a movie that you have seen. (Put movie names in quotation marks)"
      elif len(matches) >= 2:
        response = "Please tell me about one movie at a time. Go ahead."
      elif matches[0] == "":
        response = "You put a blank movie!"
      else:
        movie_name = matches[0]
        processed_sentence = self.remove_movie(input)


      return movie_name, response, processed_sentence

    def remove_movie(self, input):
      input_array = input.split('"')
      return input_array[0]+input_array[-1]

    def process(self, input):
      """Takes the input string from the REPL and call delegated functions
      that
        1) extract the relevant information and
        2) transform the information into a response to the user
      """
      #############################################################################
      # TODO: Implement the extraction and transformation in this method, possibly#
      # calling other functions. Although modular code is not graded, it is       #
      # highly recommended                                                        #
      #############################################################################

      response = ""

      if self.is_turbo == True:
        pass
      else:
        movie_name, response, processed_sentence = self.extract_movie(input)
        if movie_name == "":
          return response
        else:
          sentiment = self.get_sentence_sentiment(processed_sentence)
          if sentiment != 0:
            self.movies[movie_name] = sentiment
            if sentiment == 1:
              response = 'You liked "'+movie_name+'". Thank you!'
            else:
              response = 'You did not like "'+movie_name+'". Thank you!'
          else:
            return "Sorry, I don't understand. Did you like or dislike this movie?"

      if len(self.movies) == 5:
        print(response)
        print("Recommending movie now")
        return ""
      else:
        response += "\nTell me about another movie you have seen."
        return response

    def get_sentence_sentiment(self, sentence):
      sentence = sentence.split()
      not_flags = self.flagWords(sentence, self.negateWords)
      not_prev_flags = self.flagWordsBackwards(sentence, self.prevNegateWords)
      extreme_flags = self.flagWords(sentence, self.extremeWords)
      pos_count = 0
      neg_count = 0
      for word_index in range(len(sentence)):
        word = self.p.stem(sentence[word_index])
        if word in self.sentiment:
          if not_flags[word_index] == 0 and not_prev_flags[word_index] == 0:
            if self.sentiment[word] == "pos":
              pos_count += 1
            else:
              neg_count += 1
          else:
            if self.sentiment[word] == "pos":
              neg_count += 1
            else:
              pos_count += 1
      if pos_count > neg_count:
        return 1
      elif neg_count > pos_count:
        return -1
      else:
        return 0


    #############################################################################
    # 3. Movie Recommendation helper functions                                  #
    #############################################################################

    def read_data(self):
      """Reads the ratings matrix from file"""
      # This matrix has the following shape: num_movies x num_users
      # The values stored in each row i and column j is the rating for
      # movie i by user j
      self.titles, self.ratings = ratings()
      reader = csv.reader(open('data/sentiment.txt', 'rb'))
      self.sentiment = dict(reader)
      self.stem_words()
      self.binarize()
      print(self.titles)

    def make_movie_to_index_dict(self):
      for word_index in range(len(self.titles)):
        movie_index = self.titles[word_index][0]

    def stem_words(self):
      new_sentiment = {}
      for word in self.sentiment:
        new_sentiment[self.p.stem(word)] = self.sentiment[word]
      self.sentiment = new_sentiment

    def binarize(self):
      """Modifies the ratings matrix to make all of the ratings binary"""
      for movie_index in range(self.ratings.shape[0]):
        for user_index in range(self.ratings.shape[1]):
          rating = self.ratings[movie_index, user_index]
          if rating >= 2.5:
            self.ratings[movie_index, user_index] = 1
          elif rating > 0:
            self.ratings[movie_index, user_index] = -1


    def distance(self, u, v):
      """Calculates a given distance function between vectors u and v"""
      # TODO: Implement the distance function between vectors u and v]
      # Note: you can also think of this as computing a similarity measure
      return np.dot(u, v)/(np.linalg.norm(u)*np.linalg.norm(v))


    def recommend(self, u):
      """Generates a list of movies based on the input vector u using
      collaborative filtering"""
      # TODO: Implement a recommendation function that takes a user vector u
      # and outputs a list of movies recommended by the chatbot
      # for user_movies in self.movies:
      #   for movie_index in 
      pass

    def flagWords(self, words, mySet):
      currentWordIndex = 0
      prefix_list = [0 for x in range(len(words))]
      while currentWordIndex < len(words):
        currentWord = words[currentWordIndex]
        if currentWord in mySet:
          for currentPrefixIndex in range(currentWordIndex+1, len(words)):
            if words[currentPrefixIndex] in self.punctuation or words[currentPrefixIndex] in mySet:
              break
            prefix_list[currentPrefixIndex] = 1 if prefix_list[currentPrefixIndex] == 0 else 0
        currentWordIndex += 1
      return prefix_list

    def flagWordsBackwards(self, words, mySet):
      currentWordIndex = len(words)-1
      prefix_list = [0 for x in range(len(words))]
      while currentWordIndex >= 0:
        currentWord = words[currentWordIndex]
        if currentWord in mySet:
          for currentPrefixIndex in range(currentWordIndex-1, -1, -1):
            if words[currentPrefixIndex] in self.endPunctuation or words[currentPrefixIndex] in mySet:
              break
            prefix_list[currentPrefixIndex] = 1 if prefix_list[currentPrefixIndex] == 0 else 0
        currentWordIndex -= 1
      return prefix_list

    #############################################################################
    # 4. Debug info                                                             #
    #############################################################################

    def debug(self, input):
      """Returns debug information as a string for the input string from the REPL"""
      # Pass the debug information that you may think is important for your
      # evaluators
      debug_info = 'debug info'
      return debug_info


    #############################################################################
    # 5. Write a description for your chatbot here!                             #
    #############################################################################
    def intro(self):
      return """
      Your task is to implement the chatbot as detailed in the PA6 instructions.
      Remember: in the starter mode, movie names will come in quotation marks and
      expressions of sentiment will be simple!
      Write here the description for your own chatbot!
      """


    #############################################################################
    # Auxiliary methods for the chatbot.                                        #
    #                                                                           #
    # DO NOT CHANGE THE CODE BELOW!                                             #
    #                                                                           #
    #############################################################################

    def bot_name(self):
      return self.name

if __name__ == '__main__':
    Chatbot()
