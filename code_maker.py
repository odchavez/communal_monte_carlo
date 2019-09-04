import random
import string

class randomString:
    
    def __init__(self, stringLength=10):
        """Generate a random string of fixed length """
        letters = string.ascii_lowercase
        self.code =  "".join(random.choice(letters) for i in range(stringLength))
