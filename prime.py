def is_prime(num):
    if num <= 1:
        return False
    if num <= 3:
        return True
    if num % 2 == 0 or num % 3 == 0:
        return False
    i = 5
    while i * i <= num:
        if num % i == 0 or num % (i + 2) == 0:
            return False
        i += 6
    return True

# Generate prime numbers between 1 and 100
prime_numbers = [str(num) for num in range(1, 100) if is_prime(num)]

# Save the prime numbers to a file
with open("prime.txt", "w") as file:
    file.write("\n".join(prime_numbers))
    
print("Prime numbers between 1 and 100 saved to prime.txt")