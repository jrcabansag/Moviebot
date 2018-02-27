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
import random

import numpy as np

from movielens import ratings
from random import randint
from PorterStemmer import PorterStemmer


""" TO-DO:

COMPLETED FEATURES:
- alternate titles
- not binarizing (maybe)

FEATURES:
- figure out how to choose to not binarize (turbo?? wtF??)
- arbitrary input
- multiple movies
- disambiguating movie titles for series
- emotion detection
- spellcheck (edit distance)
- extreme like/dislike
"""
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
      self.negateWords = set(["not", "no", "never", "neither", "nor"])
      self.prevNegateWords = set(["but", "although", "however", "yet"])
      self.extremeWords = set(["very", "really", "extremely"])
      self.movies = {}
      self.movie_to_index_dict = {}
      self.alternate_titles_dict = {}
      self.movie_scores = []
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

    def handle_prefixes(self, movie_name, prefix_list):
      for prefix_index in range(len(prefix_list)):
        prefix = prefix_list[prefix_index]
        year_patt = '\(\d\d\d\d\)$'
        matches = re.findall(year_patt, movie_name)
        #print("Matches for {} is {}".format(movie_name, matches))
        if len(matches) == 0 and movie_name[0:len(prefix)+1] == prefix+" ":
          movie_name = movie_name[len(prefix)+1:]+", "+prefix
          return movie_name
        elif movie_name[0:len(prefix)+1] == prefix+" ":
          movie_year = movie_name[len(movie_name)-7:]
          movie_name = movie_name[len(prefix)+1:-7]+", "+prefix+movie_year
          return movie_name
      return movie_name

    def remove_year(self, movie_title_with_year):
      return movie_title_with_year[0:len(movie_title_with_year)-7]

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
        movie_name = self.handle_prefixes(movie_name, ["The", "A", "La", "El"])
        print("SEARCHING FOR {}".format(movie_name))
        if movie_name not in self.movie_to_index_dict:
          if self.is_turbo and movie_name.lower() in self.alternate_titles_dict:
            movie_name =  self.alternate_titles_dict[movie_name.lower()]
          elif self.is_turbo and self.remove_year(movie_name).lower() in self.alternate_titles_dict:
            movie_name = self.alternate_titles_dict[self.remove_year(movie_name).lower()]
          else:
            movie_name = ""
            response = "That is not a valid movie!"
        if movie_name in self.movies:
          movie_name = ""
          response = "You already talked about that movie! Tell me about other movies."
        processed_sentence = self.remove_movie(input)


      return movie_name, response, processed_sentence

    def remove_movie(self, input):
      input_array = input.split('"')
      return input_array[0]+input_array[-1]

    def check_if_yes(self, input):
      return input == "Yes"

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
      if len(self.movies) == 5:
        if self.check_if_yes(input):
          return "I recommend "+self.get_top_recommendation()+"\n Would you like to hear another recommendation? (Or enter :quit if you're done.)"
        elif input == ":quit":
          return ""
        else:
          return "Sorry, I don't understand, type Yes if you would like to hear another recommendation? (Or enter :quit if you're done.)"
      else:
        movie_name, response, processed_sentence = self.extract_movie(input)
        if movie_name == "":
          return response
        sentiment = self.get_sentence_sentiment(processed_sentence)
        if sentiment != 0:
          self.movies[movie_name] = sentiment
          if sentiment == 1:
            like_greeting_one = ["Glad to hear you liked "+movie_name, "Happy to hear you enjoyed "+movie_name, "You liked "+movie_name, movie_name+" sounds like a great movie"]
            like_greeting_two = [". Thanks! ", ". Thank you! ", ". Neato! ", ". Sounds cool! "]
            response = like_greeting_one[random.randrange(len(like_greeting_one))]+like_greeting_two[random.randrange(len(like_greeting_two))]
          else:
            dislike_greeting_one = ["Sorry you didn't like "+movie_name, "You didn't like "+movie_name, "So you didn't enjoy "+movie_name, "So "+movie_name+" wasn't the best movie in your opinion"]
            dislike_greeting_two = [". Thanks for letting me know! ", ". Thanks! ", ". I understand. ", ". Alright. "]
            response = dislike_greeting_one[random.randrange(len(dislike_greeting_one))]+dislike_greeting_two[random.randrange(len(dislike_greeting_two))]
          if len(self.movies) == 5:
            self.movie_scores = self.recommend()
            response += "I recommend "+self.get_top_recommendation()+"\n Would you like to hear another recommendation? (Or enter :quit if you're done.)"
          else:
            response += "\nTell me about another movie you have seen."
        else:
          return "Sorry, I don't understand: "+input+". Did you like or dislike this movie?"
        return response

    def get_top_recommendation(self):
      top_movie = np.argmax(self.movie_scores)
      self.movie_scores[top_movie] = -sys.maxint-1
      return self.get_movie_title(self.titles[top_movie][0])


    def get_sentence_sentiment(self, sentence):
      sentence = sentence.split()
      not_flags = self.flagWords(sentence, self.negateWords)
      not_prev_flags = self.flagWordsBackwards(sentence, self.prevNegateWords)
      #extreme_flags = self.flagWords(sentence, self.extremeWords)
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
      self.make_alternate_titles_dict()
      reader = csv.reader(open('data/sentiment.txt', 'rb'))
      self.sentiment = dict(reader)
      self.stem_words()
      self.binarize()
      self.make_movie_to_index_dict()

    def make_alternate_titles_dict(self):
      for movie_index in range(len(self.titles)):
        movie_title = self.titles[movie_index][0]
        movie_patt = '\((.*?)\)'
        matches = re.findall(movie_patt, movie_title)
        for match_index in range(len(matches)-1):
          match = matches[match_index]
          if match[0:6] == "a.k.a.":
            match = match[7:]
          self.alternate_titles_dict[match.lower()] = movie_title
        self.alternate_titles_dict[self.remove_year(movie_title).lower()] = movie_title
        self.alternate_titles_dict[movie_title.lower()] = movie_title


    def make_movie_to_index_dict(self):
      for movie_index in range(len(self.titles)):
        movie_title = self.get_movie_title(self.titles[movie_index][0])
        self.movie_to_index_dict[movie_title] = movie_index

    def get_movie_title(self, movie_title_with_year):
      return movie_title_with_year

    def stem_words(self):
      new_sentiment = {}
      for word in self.sentiment:
        new_sentiment[self.p.stem(word)] = self.sentiment[word]
      self.sentiment = new_sentiment

    def binarize(self):
      """Modifies the ratings matrix to make all of the ratings binary"""
      negative_coordinates_row, negative_coordinates_col = np.where(self.ratings > 0)
      positive_coordinates_row, positive_coordinates_col = np.where(self.ratings >= 2.5)
      print(negative_coordinates_col)
      for x in range(len(negative_coordinates_row)):
        self.ratings[negative_coordinates_row[x], negative_coordinates_col[x]] = -1
      for x in range(len(positive_coordinates_row)):
        self.ratings[positive_coordinates_row[x], positive_coordinates_col[x]] = 1

    def distance(self, u, v):
      """Calculates a given distance function between vectors u and v"""
      # TODO: Implement the distance function between vectors u and v]
      # Note: you can also think of this as computing a similarity measure
      dot_product = np.dot(u, v)
      if dot_product == 0:
        return 0
      else:
        return dot_product/(np.linalg.norm(u)*np.linalg.norm(v))

    def pearson(self, u, v):
      """Calculates a given distance function between vectors u and v"""
      # TODO: Implement the distance function between vectors u and v]
      # Note: you can also think of this as computing a similarity measure
      u = np.copy(u)
      v = np.copy(v)
      u_rated = np.where(u != 0)[0]
      #print(u_rated)
      for u_index in u_rated:
        #print("Rating in index {} is {}".format(u_index, u[u_index]))
        u[u_index] = u[u_index] - np.average(u)
        #print("Rating in index {} is now {}".format(u_index, u[u_index]))
      v_rated = np.where(v != 0)[0]
      for v_index in v_rated:
        v[v_index] = v[v_index] - np.average(v)
      dot_product = np.dot(u, v)
      if dot_product == 0:
        return 0
      else:
        return dot_product/(np.linalg.norm(u)*np.linalg.norm(v))


    def recommend(self):
      """Generates a list of movies based on the input vector u using
      collaborative filtering"""
      # TODO: Implement a recommendation function that takes a user vector u
      # and outputs a list of movies recommended by the chatbot
      movie_scores = []
      for movie_index in range(len(self.ratings)):
        score = 0
        for user_movie in self.movies:
          user_movie_index = self.movie_to_index_dict[user_movie]
          score += self.distance(self.ratings[movie_index], self.ratings[user_movie_index])*self.movies[user_movie]  
        movie_scores.append(score)
      movie_scores = np.array(movie_scores)
      for user_movie in self.movies:
        user_movie_index = self.movie_to_index_dict[user_movie]
        movie_scores[user_movie_index] = -sys.maxint-1
      return movie_scores

    def flagWords(self, words, mySet):
      currentWordIndex = 0
      prefix_list = [0 for x in range(len(words))]
      while currentWordIndex < len(words):
        currentWord = words[currentWordIndex]
        if currentWord in mySet or currentWord[-3:] == "n't":
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
