{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "2dfe2bf6",
   "metadata": {},
   "outputs": [],
   "source": [
    "import streamlit as st\n",
    "st.text(\"Hello World\")\n",
    "original_title='<p style=\"font-family:Courier; color:Blue; font-size: 20px;\">Hello World</p>'\n",
    "st.markdown(original_title,unsafe_allow_html=True)\n",
    "new_title='<p style=\"font-family:sans-serif; color:Green; font-size: 42px;\">Hello World</p>'\n",
    "st.markdown(new_title,unsafe_allow_html=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "0b0d2c56",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
