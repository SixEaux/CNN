from collections.abc import Callable

def conversation_save(func:Callable, save_to:str, minus_y:bool, type_thing:str="thing"):
    if save_to != "":

        if minus_y:
            func()

        else:

            while True:

                i = input(f"Are you sure you want to save this {type_thing}? (y/n)")

                if i == "y":
                    func()
                    break

                elif i == "n":
                    print("You decided not to save.")
                    break
            
                elif i == "oh no an infinite loop":
                    print("Don't worry, I am here")
                    break
            
                else:
                    print("Not a valid input.")