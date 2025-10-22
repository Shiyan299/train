# import packages
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


# show the title
def main():
	print('Titanic EDA - Fare boxplots by Passenger Class')

	# read csv and show the dataframe
	csv_path = 'train.csv'
	if not os.path.exists(csv_path):
		raise FileNotFoundError(f"Could not find {csv_path} in the current directory: {os.getcwd()}")

	df = pd.read_csv(csv_path)
	print('\nDataframe head:')
	print(df.head())

	# create a figure with three subplots, size should be (15, 5)
	# show the box plot for ticket price with different classes
	# you need to set the x labels and y labels
	sns.set_style('whitegrid')
	plt.figure(figsize=(15, 5))

	# We'll plot Fare distribution for each Pclass in separate subplots
	for i, pclass in enumerate([1, 2, 3], start=1):
		ax = plt.subplot(1, 3, i)
		subset = df[df['Pclass'] == pclass]
		sns.boxplot(x='Pclass', y='Fare', data=subset, ax=ax)
		ax.set_title(f'Pclass {pclass}')
		ax.set_xlabel('Passenger Class')
		ax.set_ylabel('Fare ($)')

	plt.tight_layout()
	out_file = 'fare_boxplots.png'
	plt.savefig(out_file)
	print(f"Saved figure to {out_file}")


if __name__ == '__main__':
	main()

