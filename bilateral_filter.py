# bilateral filter
'''def gaussian(x, sigma):
    return (1.0 / (2 * np.pi * (sigma ** 2))) * np.exp(-(x ** 2) / (2 * (sigma ** 2)))


def distance(x1, y1, x2, y2):
    return np.sqrt(np.abs((x1 - x2) ** 2 - (y1 - y2) ** 2))


def bilateral_filter(image, diameter, sigma_i, sigma_s):
    new_image = np.zeros(image.shape)

    for row in range(len(image)):
        for col in range(len(image[0])):
            wp_total = 0
            filtered_image = 0
            for k in range(diameter):
                for l in range(diameter):
                    n_x = row - (diameter / 2 - k)
                    n_y = col - (diameter / 2 - l)
                    if n_x >= len(image):
                        n_x -= len(image)
                    if n_y >= len(image[0]):
                        n_y -= len(image[0])
                    gi = gaussian(image[int(n_x)][int(n_y)] - image[row][col], sigma_i)
                    gs = gaussian(distance(n_x, n_y, row, col), sigma_s)
                    wp = gi * gs
                    filtered_image = filtered_image + (image[int(n_x)][int(n_y)] * wp)
                    wp_total = wp_total + wp
            filtered_image = filtered_image // wp_total
            new_image[row][col] = int(np.round(filtered_image))
    return new_image


dc_bilateral = bilateral_filter(dc, 7, 20.0, 20.0)

fig, ax = plt.subplots(1, 2, figsize=(20, 12))
ax[0].imshow(dc, cmap='gray', interpolation=inter)
ax[0].set_title('Original')

ax[1].imshow(dc_bilateral, cmap='gray', interpolation=inter)
ax[1].set_title('bilateral')
plt.show()'''
