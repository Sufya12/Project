#!/usr/bin/env python
# coding: utf-8

# In[1]:


print("Welcome to Treasure Isiand ")
print("Your mission is to find the treasure")
want_go = str(input("""You're at a cross road. Where do you want to go? Type "left" or "right" """))
if want_go == "left":
    wait_swim = str(input("""Tou come to a lake. There is an island in the middle of the lake. Type "wait" to wait for a boat.Type "swim " to swim across.\n """))
    if wait_swim == "wait":
        which_door = str(input("""Which Door are Use. "red" , "blue" , "yellow"  """))
        if which_door == "yellow":
            print("You Win.")
        else:
            print("Game Over.")

    else:
        print("Game Over.")
else:
    print("Game Over.")


# In[ ]:




