# Copyright 2020-2022 antillia.com Toshiyuki Arai
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# 2022/09/05 Updated.

#  PILImageCropper.py

# encodig: utf-8

import sys
import os
import traceback
import numpy as np
from PIL import Image

#---------------------------------------------------------------------

class PILImageCropper():

  def __init__(self, filename =None):
    self.image = None #Pillow image
    self.filename = filename
    if self.filename != None:
      #  Load an image from the filaname as Pillow image format.  
      self.image = Image.open(filename)
      

  def crop_maximum_square_region(self, cropped_filename):
    if self.image != None:
      # Crop max square region from the self.image, and save it as a cropped_file.
      w, h  = self.image.size
      ms    = min([w, h])
      self.cropped_image = self.image.crop(((w - ms) // 2, (h - ms) // 2,
                                       (w + ms) // 2, (h + ms) // 2))

      self.cropped_image.save(cropped_filename)

 
  def crop_largest_central_square_region(self):
    if self.image != None:
      # Crop max square region from the self.image, and save it as a cropped_file.
      w, h  = self.image.size
      ms    = min([w, h])
      self.cropped_image = self.image.crop(((w - ms) // 2, (h - ms) // 2,
                                       (w + ms) // 2, (h + ms) // 2))

      return self.cropped_image
      

 
