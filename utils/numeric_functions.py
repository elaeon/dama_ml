def pearsoncc(X, Y):
   """ Compute Pearson Correlation Coefficient. """
   X = (X - X.mean(0)) / X.std(0)
   Y = (Y - Y.mean(0)) / Y.std(0)
   return (X * Y).mean()
