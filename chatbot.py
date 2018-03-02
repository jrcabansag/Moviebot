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
- not binarizing
- movies without quotes (require caps on first word)
- emotion detection
- extreme like/dislike
- arbitrary input

FEATURES:
- multiple movies
- disambiguating movie titles for series
- spellcheck (edit distance)
"""
class Chatbot:
    """Simple class to implement the chatbot for PA 6."""

    #############################################################################
    # `moviebot` is the default chatbot. Change it to your chatbot's name       #
    #############################################################################
    def __init__(self, is_turbo=False):
      self.name = 'Moviebot'
      self.is_turbo = is_turbo
      self.is_binarized = True
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
      self.non_binarized_matrix = {}
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
      year_patt = '(\(\d\d\d\d.*?\))'
      year_matches = re.findall(year_patt, movie_title_with_year)
      if len(year_matches) != 0:
        return movie_title_with_year[0:len(movie_title_with_year)-len(year_matches[0])-1]
      else:
        return movie_title_with_year

    def is_capitalized(self, word):
      if word == "":
        return False
      else:
        return word[0].isupper() or word[0].isdigit()

    def find_non_quote_title(self, input):
      year_patt = '(\(\d\d\d\d.*?\))'
      year_matches = re.findall(year_patt, input)
      word_array = input.split()
      if len(year_matches) != 0:
        year_index = -1
        for word_index in range(len(word_array)):
          if word_array[word_index].find(year_matches[0]) != -1:
            year_index = word_index
            word_array[year_index] = year_matches[0]
        for word_index in range(year_index):
          if self.is_capitalized(word_array[word_index]):
            temp_movie_title = " ".join(word_array[word_index:year_index]) # I liked Full Sequence (2014)
            temp_movie_title = self.handle_prefixes(temp_movie_title, ["The", "A", "La", "El"])
            temp_movie_title = self.process_for_alternate_dict(temp_movie_title)
            #print("CHECKING TITLE {}".format(temp_movie_title))
            if temp_movie_title in self.alternate_titles_dict:
              return temp_movie_title, self.alternate_titles_dict[temp_movie_title]
        return "", ""
      else:
        #print("DIDNT FIND YEAR")
        for capital_word_index in range(len(word_array)):
          if self.is_capitalized(word_array[capital_word_index]):
            #print("CAPITALIZED WORD IS {}".format(word_array[capital_word_index]))
            for right_word_index in range(len(word_array), capital_word_index-1, -1):
              temp_movie_title = " ".join(word_array[capital_word_index:right_word_index+1])
              temp_movie_title = self.handle_prefixes(temp_movie_title, ["The", "A", "La", "El"])
              temp_movie_title = self.process_for_alternate_dict(temp_movie_title)
              #print("CHECKING TITLE {}".format(temp_movie_title))
              if temp_movie_title in self.alternate_titles_dict:
                return temp_movie_title, self.alternate_titles_dict[temp_movie_title]
        return "", ""

    def process_for_alternate_dict(self, input):
      for char in ".!?,":
        input = input.replace(char, "")
      return input.lower()

    def remove_punctuation(self, input):
      for char in ".!?,":
        input = input.replace(char, "")
      return input

    def find_in_input(self, input, input_list, response_list):
      input = input.lower()
      for index in range(len(input_list)):
        matches = re.findall("(?:\W|^)"+input_list[index]+"(?:\W|$)", input)
        if len(matches) != 0:
          return response_list[random.randrange(0, len(response_list))]+" "
      return ""

    def find_regex_in_input(self, input, input_list, response_list):
      input = input.lower()
      for index in range(len(input_list)):
        matches = re.findall("(?:\W|^)"+input_list[index], input)
        if len(matches) != 0:
          return response_list[random.randrange(0, len(response_list))]+matches[0]+". "
      return ""

    def check_if_arbitrary(self, input):
      response = ""
      hi_inputs = ["hello", "hi", "sup", "hey"]
      hi_responses = ["Hello!", "Hi!", "Sup!", "Hey!"]
      identity_inputs = ["who are you", "what are you", "your name"]
      identity_responses = ["I'm "+self.name+"!", self.name+" is my name, recommending movies is my game."]
      how_are_you_inputs = ["how are you", "what's up", "how's it going", "how's your"]
      how_are_you_responses = ["I'm doing good!", "I'm great!", "Things could be better."]
      topic_inputs = ["i want to", "let's", "can we", "can you"]
      topic_input_suffix = "(.*?)(?:\.|$|,|!|because|cause)"
      topic_inputs = [x+topic_input_suffix for x in topic_inputs]
      topic_responses = ["I don't really want to", "Let's not"]
      existential_question = ["what's your", "do you have a", "do you think about"]
      talk_about_movies_responses = ["Let's talk about movies! ", "We should discuss movies! ", "Tell me about a movie you saw! "]
      response += self.find_in_input(input, hi_inputs, hi_responses)
      response += self.find_in_input(input, identity_inputs, identity_responses)
      response += self.find_in_input(input, how_are_you_inputs, how_are_you_responses)
      regex_reponse = self.find_regex_in_input(input, topic_inputs, topic_responses)
      if regex_reponse != "":
        regex_reponse += talk_about_movies_responses[random.randrange(0, len(talk_about_movies_responses))]
      response += regex_reponse


      return response


    def extract_movie(self, input):
      movie_patt = '\"(.*?)\"'
      matches = re.findall(movie_patt, input)
      movie_name = ""
      response = ""
      processed_sentence = input.lower()
      if len(matches) == 0:
        if self.is_turbo == False:
          response = "Tell me about a movie that you have seen. (Put movie names in quotation marks)"
        else:
          temp_movie_name, movie_name = self.find_non_quote_title(input)
          processed_sentence = processed_sentence.replace(temp_movie_name, "")
          if movie_name == "":
            response = "Tell me about a movie that you have seen."
      elif len(matches) >= 2:
        response = "Please tell me about one movie at a time. Go ahead."
      elif matches[0] == "":
        response = "You put a blank movie!"
      else:
        movie_name = matches[0]
        movie_name = self.handle_prefixes(movie_name, ["The", "A", "La", "El"])
        #print("SEARCHING FOR {}".format(movie_name))
        if movie_name not in self.movie_to_index_dict:
          movie_name = self.process_for_alternate_dict(movie_name)
          if self.is_turbo and movie_name in self.alternate_titles_dict:
            movie_name =  self.alternate_titles_dict[movie_name]
          elif self.is_turbo and self.remove_year(movie_name) in self.alternate_titles_dict:
            movie_name = self.alternate_titles_dict[self.remove_year(movie_name)]
          else:
            movie_name = ""
            response = "That is not a valid movie!"
        processed_sentence = self.remove_movie(input.lower())
        if movie_name in self.movies:
          movie_name = ""
          response = "You already talked about that movie! Tell me about other movies."
        #processed_sentence = self.remove_movie(input)


      return movie_name, response, processed_sentence

    def remove_movie(self, input):
      input_array = input.split('"')
      return input_array[0]+input_array[-1]

    def check_if_yes(self, input):
      yes_list = ["yes", "y", "yeah", "sure", "ok", "okay", "k"]
      for index in range(len(yes_list)):
        yes_matches = re.findall("(?:\W|^)"+yes_list[index]+"(?:\W|$)", input.lower())
        if len(yes_matches) != 0:
          return True
      return False

    def find_emotion(self, input):
      feeling_contexts = ["I am", "I'm", "feel", "me", "felt", "I was", "You are", "This is", "Feeling"]
      amplifiers = ["so", "very", "really", "extremely", "super", ""]
      adjectives_set = ["depressed", "sad", "angry", "furious", "happy", "annoyed", "frustrated", "bored", "confused", "excited", "overjoyed", "amazed", "inspired", "mad", "tired", "hungry", "glad"]
      verbs_set = ["cry", "laugh"]
      exclamation = ""
      if input[-1] == "!":
        exclamation = "really "
      for feeling_index in range(len(feeling_contexts)):
        for amplifier_index in range(len(amplifiers)):
          feelings_matches = re.findall(feeling_contexts[feeling_index].lower()+" (?:"+amplifiers[amplifier_index]+" ?)+ ?(\w*) ?", input.lower())
          for feelings_match_index in range(len(feelings_matches)):
            if amplifier_index != len(amplifiers)-1:
              exclamation = "really "
            for adjectives_index in range(len(adjectives_set)):
              stemmed_feeling = self.p.stem(feelings_matches[feelings_match_index])
              if stemmed_feeling == self.p.stem(adjectives_set[adjectives_index]):
                if self.sentiment[stemmed_feeling] == "pos":
                  return "Glad to hear you're feeling "+exclamation+adjectives_set[adjectives_index]+". "
                else:
                  return "Sorry to hear you're feeling "+exclamation+adjectives_set[adjectives_index]+". "
            for verbs_index in range(len(verbs_set)):
              stemmed_feeling = self.p.stem(feelings_matches[feelings_match_index])
              if stemmed_feeling == self.p.stem(verbs_set[verbs_index]):
                if self.sentiment[stemmed_feeling] == "pos":
                  return "Well, it's good to "+verbs_set[verbs_index]+" :) "
                else:
                  return "Sorry to hear that, it's good to "+verbs_set[verbs_index]+" once in a while. "
      return ""

    def has_fine_grained_sentiment(self, input):
      input = input.replace("didn't really like", "didn't like")
      input = input.replace("wasn't really", "wasn't")
      input = input.replace("wasn't very", "wasn't")
      input = input.replace("wasn't super", "wasn't")
      trigger_words = ["love", "hate", "amazing", "terrible", "really", "very", "super", "horrible", "excellent", "!", "favorite", "best", "worst"]
      for word_index in range(len(trigger_words)):
        trigger_matches = re.findall(trigger_words[word_index], input.lower())
        if len(trigger_matches) != 0:
          return True
      return False

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
      if self.is_turbo == True and self.is_binarized == True:
        #print("NOT BINARIZING MATRIX")
        self.is_binarized = False
        self.ratings = self.non_binarized_matrix
      response = ""
      if self.is_turbo == True:
        response = self.find_emotion(input)
        #print("FIND EMOTION RETURNED {}".format(response))
      if len(self.movies) == 5:
        if self.check_if_yes(input):
          return "I recommend "+self.get_top_recommendation()+"\n Would you like to hear another recommendation? (Or enter :quit if you're done.)"
        elif input == ":quit":
          return ""
        else:
          return "Sorry, I don't understand, type Yes if you would like to hear another recommendation (Or enter :quit if you're done.)"
      else:
        movie_name, temp_response, processed_sentence = self.extract_movie(input)
        if movie_name == "" and self.is_turbo == False:
          return "Sorry I don't understand, let's talk about movies!"
        elif movie_name == "" and self.is_turbo == True:
          arbitrary_response = self.check_if_arbitrary(input)
          if arbitrary_response == "" and response == "":
            return response+"Sorry, I don't understand. "+temp_response
          else:
            return response+arbitrary_response
        elif movie_name == "" and response != "":
          return response+temp_response
        #print(processed_sentence)
        sentiment = self.get_sentence_sentiment(processed_sentence)
        has_fine_grained_sentiment = self.has_fine_grained_sentiment(processed_sentence)
        fine_grained_sentiment = ""
        if has_fine_grained_sentiment == True and self.is_turbo:
          fine_grained_sentiment = "REALLY "
        if sentiment != 0:
          self.movies[movie_name] = sentiment
          if sentiment == 1:
            like_greeting_one = ["Glad to hear you "+fine_grained_sentiment+"liked "+movie_name, "Happy to hear you "+fine_grained_sentiment+"enjoyed "+movie_name, "You "+fine_grained_sentiment+"liked "+movie_name, movie_name+" sounds like "+fine_grained_sentiment+"a great movie"]
            like_greeting_two = [". Thanks! ", ". Thank you! ", ". Neato! ", ". Sounds cool! "]
            response += like_greeting_one[random.randrange(len(like_greeting_one))]+like_greeting_two[random.randrange(len(like_greeting_two))]
          else:
            dislike_greeting_one = ["Sorry you "+fine_grained_sentiment+"didn't like "+movie_name, "You "+fine_grained_sentiment+"didn't like "+movie_name, "So you "+fine_grained_sentiment+"didn't enjoy "+movie_name, "So "+movie_name+" "+fine_grained_sentiment+"wasn't the best movie in your opinion"]
            dislike_greeting_two = [". Thanks for letting me know! ", ". Thanks! ", ". I understand. ", ". Alright. "]
            response += dislike_greeting_one[random.randrange(len(dislike_greeting_one))]+dislike_greeting_two[random.randrange(len(dislike_greeting_two))]
          if len(self.movies) == 5:
            self.movie_scores = self.recommend()
            response += "I recommend "+self.get_top_recommendation()+"\n Would you like to hear another recommendation? (Or enter :quit if you're done.)"
          else:
            response += "\nTell me about another movie you have seen."
        else:
          return "Sorry, I don't understand. Did you like or dislike this movie?"
        return response

    def get_top_recommendation(self):
      top_movie = np.argmax(self.movie_scores)
      self.movie_scores[top_movie] = -sys.maxint-1
      return self.get_movie_title(self.titles[top_movie][0])


    def get_sentence_sentiment(self, sentence):
      sentence = self.remove_punctuation(sentence)
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
      #print(self.alternate_titles_dict)
      reader = csv.reader(open('data/sentiment.txt', 'rb'))
      self.sentiment = dict(reader)
      self.stem_words()
      self.non_binarized_matrix = np.copy(self.ratings)
      self.binarize()
      self.make_movie_to_index_dict()

    def make_alternate_titles_dict(self): #a.k.a without the year, anything in parenthesis without the year, full movie title without the year, movie title itself
      for movie_index in range(len(self.titles)):
        movie_title = self.titles[movie_index][0]
        movie_patt = '\((.*?)\)'
        year_patt = '(\(\d\d\d\d.*?\))'
        year_matches = re.findall(year_patt, movie_title)
        if len(year_matches) == 0:
          year = ""
        else:
          year = year_matches[0]
        matches = re.findall(movie_patt, movie_title)
        for match_index in range(len(matches)-1):
          match = matches[match_index]
          if match[0:6] == "a.k.a.":
            match = match[7:]
          match = self.process_for_alternate_dict(match)
          self.alternate_titles_dict[match] = movie_title
          self.alternate_titles_dict[match+" "+year] = movie_title
        temp_movie_title = self.process_for_alternate_dict(movie_title)
        self.alternate_titles_dict[self.remove_year(temp_movie_title)] = movie_title
        self.alternate_titles_dict[temp_movie_title] = movie_title


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
      #print(negative_coordinates_col)
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
      Turbo Moviebot loves talking about movies! He can:
      - identify movies without quotation marks or perfect capitalization
      - understand fine-grained sentiment extraction
      - identify and respond to emotions
      - respond to arbitrary input
      - use a non binarized dataset
      - understand alternate/foreign titles.
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
