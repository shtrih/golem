from PIL import Image, ImageFilter, ImagePalette, ImageEnhance, ImageDraw
import numpy
from typing import List, Callable
import time


class PALETTE_INDEX_MAP(object):
    """ https://stackoverflow.com/questions/2682745/how-do-i-create-a-constant-in-python
    """
    __slots__ = ()
    GROUND = 0
    UNDISCOVERED = 1
    WALL = 2


PALETTE_MAP = PALETTE_INDEX_MAP()
COLOR_PALETTE = [
    # Цвета для PALETTE_MAP
    210, 210, 210,
    0, 200, 0,
    0, 0, 0,
    # Цвета для весов
    95, 255, 55,
    125, 215, 45,
    155, 185, 35,
    185, 155, 25,
    215, 125, 15,
    255, 95, 0,
]


def increase_thickness_left_up(ndarray: numpy.ndarray) -> numpy.ndarray:
    """
    Увеличиваем толщину всех линий слева и сверху

    :param ndarray:
    :return ndarray:
    """
    leni = len(ndarray)
    lenj = len(ndarray[0])
    for i in range(leni):
        for j in range(lenj):
            if ndarray[i][j] == PALETTE_INDEX_MAP.GROUND:
                try:
                    if ndarray[i + 1][j] == ndarray[i][j + 1] == ndarray[i + 1][j + 1]:
                        ndarray[i][j] = ndarray[i + 1][j]

                    if ndarray[i][j] == PALETTE_INDEX_MAP.GROUND:
                        ndarray[i][j] = ndarray[i][j + 1]
                except IndexError as e:
                    pass

            ii = leni - i - 1
            jj = lenj - j - 1
            if ndarray[ii][jj] == PALETTE_INDEX_MAP.GROUND:
                try:
                    if ndarray[ii - 1][jj] == ndarray[ii][jj - 1] == ndarray[ii - 1][jj - 1]:
                        ndarray[ii][jj] = ndarray[ii - 1][jj]
                except IndexError as e:
                    pass
            if ndarray[ii][jj] == PALETTE_INDEX_MAP.GROUND and not ndarray[ii][jj - 1] is None:
                ndarray[ii][jj] = ndarray[ii][jj - 1]

    return ndarray


def set_weight_idxs(ndarray: numpy.ndarray):
    """ Расставляем веса для алгоритма поиска пути (https://github.com/zephirdeadline/astar_python).
        Чем ближе к «стене», тем больше вес.
        Веса также выполняют роль индекса палитры, когда матрица конвертируется в картинку,
        поэтому минимальный вес у нас будет 3, ибо первые 3 индекса уже заняты (см. PALETTE_MAP).
     """
    max_distance = 2
    initial_weight = 3
    palette_weight_offset = 3 # PALETTE_MAP.WALL + 1

    def distance2weight(distance: int) -> int:
        nonlocal palette_weight_offset, max_distance, initial_weight
        return palette_weight_offset + max_distance + initial_weight - distance

    leni = len(ndarray)
    lenj = len(ndarray[0])
    for i in range(leni):
        for j in range(lenj):
            if ndarray[i][j] == PALETTE_INDEX_MAP.GROUND:
                for distance in range(1, max_distance + 1):
                    try:
                        if ndarray[i + distance][j] == PALETTE_INDEX_MAP.WALL \
                                or ndarray[i][j + distance] == PALETTE_INDEX_MAP.WALL \
                                or ndarray[i + distance][j + distance] == PALETTE_INDEX_MAP.WALL \
                                or ndarray[i + distance][j - distance] == PALETTE_INDEX_MAP.WALL:
                            ndarray[i][j] = distance2weight(distance)
                            break
                    except IndexError as e:
                        pass

            ii = leni - i - 1
            jj = lenj - j - 1
            if ndarray[ii][jj] == PALETTE_INDEX_MAP.GROUND:
                for distance in range(1, max_distance + 1):
                    try:
                        if ndarray[ii - distance][jj] == PALETTE_INDEX_MAP.WALL \
                                or ndarray[ii][jj - distance] == PALETTE_INDEX_MAP.WALL \
                                or ndarray[ii - distance][jj - distance] == PALETTE_INDEX_MAP.WALL \
                                or ndarray[ii - distance][jj + distance] == PALETTE_INDEX_MAP.WALL:
                            ndarray[ii][jj] = distance2weight(distance)
                            break
                    except IndexError as e:
                        pass
    return ndarray


def replace_weights(ndarray: numpy.ndarray) -> numpy.ndarray:
    result = numpy.where(ndarray == PALETTE_INDEX_MAP.WALL, None, ndarray)
    result = numpy.where(result == PALETTE_INDEX_MAP.UNDISCOVERED, -2, result)
    return result


def transform(image: Image, save_debug_images=False, debug_image_name='debug', dir='../images/') -> numpy.ndarray:
    image = image.quantize(64, 0)
    # http://effbot.org/zone/creating-palette-images.htm
    if save_debug_images:
        print(numpy.array(image.getpalette()).reshape((256, 3))[:10])

    def get_new_palette(old_palette: List, replace_rgb: Callable[[List[int]], List[int]]):
        j = 0
        result = []
        rgb = []

        for c in old_palette:
            rgb.append(c)
            if j == 2:
                result.extend(replace_rgb(rgb))
                j = -1
                rgb = []
            j += 1

        return result

    has_unexplored = False

    def palette_mapping(rgb: List[int]) -> List[int]:
        nonlocal has_unexplored
        # walls
        if 125 <= rgb[0] <= 155:
            if 125 <= rgb[1] <= 155:
                if 150 <= rgb[2] <= 195:
                    return [0, 0, 0]
        # unexplored area
        if 25 <= rgb[0] <= 75:
            if 130 <= rgb[1] <= 150:
                if 170 <= rgb[2] <= 190:
                    has_unexplored = True
                    return [0, 100, 0]

        return [255, 255, 255]

    new_palette = get_new_palette(image.getpalette(), palette_mapping)
    # print('has_unexplored', has_unexplored)
    if not has_unexplored:
        print('new_palette', numpy.array(new_palette).reshape((256, 3))[:10])
    image.putpalette(new_palette)

    image = image.quantize(16, 0)

    # Когда на картинке нет неисследованной области, второй цвет палитры — чёрный цвет стен.
    # Ставим цвет стен на 3 позицию, потому что позже мы маппим цвет стен именно на третью позицию в палитре
    if not has_unexplored:
        image = image.remap_palette([0, 2, 1])
    # image.putpixel((131, 130), (0, 100, 0))
    #
    # three_color_palette = [
    #     255,255,255,
    #     0,100,0,
    #     0,0,0,
    # ]
    # image.putpalette(three_color_palette)

    # Закрашиваем крестик в центре
    draw = ImageDraw.Draw(image)
    draw.rectangle([122, 122, 140, 137], PALETTE_INDEX_MAP.GROUND) # тут почему-то указывается индекс цвета в палитре
    # draw.point((130, 130), 1)

    # print('quantized', image.palette.getdata())
    # print('quantized', numpy.array(image.getpalette()).reshape((256, 3))[:5])

    # image.putpalette([
    #     0, 0, 0,
    #     # 137, 137, 188,
    #
    #     255, 255, 255,
    #     # 255, 255, 255,
    #     # 255, 0, 0,
    #     0, 0, 0,
    #     # 255, 0, 0,
    #     255, 0, 0,
    #     # 255, 255, 255,
    #     # 255, 255, 255,
    #
    #     # 255, 0, 0,
    #     # 255, 255, 0,
    #     # 255, 153, 0,
    #     # 136, 136, 187,
    #     # 137, 136, 179,
    #     # 137, 137, 188,
    #     # 130, 130, 188,
    # ])
    # print(numpy.array(image.getpalette()).reshape((256, 3))[:5])

    # image = image.quantize(13, 0)

    # print(image.getpalette())
    # image = image.convert('1')


    # (width, height) = (image.width // 2, image.height // 2)
    # image = image.resize((width, height), Image.NEAREST, None, 1)
    # image.save('images/0_filtered.gif', 'gif')
    #
    # def lookup(color):
    #     if color > 190:
    #         return 190
    #     if color > 120:
    #         return 255
    #     # if color < 16:
    #     #     return 120
    #     return 0
    # thresh = 120
    # fn = lambda x: 255 if x > thresh else 0
    # image.save('images/0_filtered.gif', 'gif')

    # image = image.convert('L').point(lookup, mode='L')
    # image = image.convert('L').point(fn, mode='L')
    # image = image.point(lookup, mode='P')
    # image = image.convert('1')
    if save_debug_images:
        image.save('{}_filtered.gif'.format(dir + debug_image_name), 'gif')

    # image = image.resize((300, 300))

    # image = image.quantize(4, 0)
    # image.putpalette([
    #     0, 0, 0,  # black background
    #     # 137, 137, 188,
    #
    #     255, 255, 255,
    #     255, 255, 255,
    #     # 255, 255, 255,
    #     # 255, 255, 255,
    #
    #     # 255, 0, 0,  # index 1 is red
    #     # 255, 255, 0,  # index 2 is yellow
    #     # 255, 153, 0,  # index 3 is orange
    #     # 136, 136, 187,
    #     # 137, 136, 179,
    #     # 137, 137, 188,
    #     # 130, 130, 188,
    # ])

    (width, height) = (image.width // 2, image.height // 2)
    image = image.resize((width, height))
    # image.save('images/0_filtered.gif', 'gif')
    ndarray = numpy.array(image)
    image.close()

    t = time.perf_counter()
    thicked = increase_thickness_left_up(ndarray)
    # numpy.savetxt("array.txt", ndarray, fmt="%s")

    # cim = Image.fromarray(thicked)
    # cim.putpalette(COLOR_PALETTE)
    # cim = cim.rotate(90)

    # ndarray = numpy.array(cim)

    elapsed_time = time.perf_counter() - t
    print('elapsed_time 1', '%.6f' % elapsed_time)

    # t = time.perf_counter()

    thicked = numpy.rot90(thicked)

    thicked = increase_thickness_left_up(thicked)
    # cim = Image.fromarray(thicked)
    # cim.putpalette(COLOR_PALETTE)
    # cim = cim.rotate(-90)
    thicked = numpy.rot90(thicked, -1)
    cim = Image.fromarray(thicked)
    cim.putpalette(COLOR_PALETTE)

    # (width, height) = (image.width // 4, image.height // 4)
    (width, height) = (image.width // 2, image.height // 2)
    cim = cim.resize((width, height))
    # cim = cim.resize((300, 300))

    ndarray = numpy.array(cim)

    t = time.perf_counter()
    ndarray = set_weight_idxs(ndarray)

    elapsed_time = time.perf_counter() - t
    print('elapsed_time 2', '%.6f' % elapsed_time)

    result = replace_weights(ndarray)

    if save_debug_images:
        numpy.savetxt('{}.txt'.format(dir + debug_image_name), result, fmt="%s")
        cim = Image.fromarray(ndarray)
        cim.putpalette(COLOR_PALETTE)
        cim.save('{}_conv.gif'.format(dir + debug_image_name), 'gif')

    cim.close()

    # ndarray = numpy.array(cim)
    # ndarray = ndarray.astype(int)
    # numpy.savetxt("array.txt", ndarray, fmt="%s")
    return result

def transform2():
    im: Image = Image.open("../images/2.gif")
    # image = Image.open("images/000.png")

    # image.filter(ImageFilter.CONTOUR)
    # image.filter(ImageFilter.EDGE_ENHANCE_MORE)
    # image.filter(ImageFilter.FIND_EDGES)
    # contrast = ImageEnhance.Contrast(image)
    # image = contrast.enhance(2)
    # sharpness = ImageEnhance.Sharpness(image)
    # image = sharpness.enhance(2)
    # image = image.convert("P")

    # image = image.quantize(64, 0)
    #
    # # http://effbot.org/zone/creating-palette-images.htm
    im.putpalette([
        0, 0, 0,  # black background
        # 137, 137, 188,

        255, 255, 255,
        255, 255, 255,
        # 255, 255, 255,
        # 255, 255, 255,

        # 255, 0, 0,  # index 1 is red
        # 255, 255, 0,  # index 2 is yellow
        # 255, 153, 0,  # index 3 is orange
        # 136, 136, 187,
        # 137, 136, 179,
        # 137, 137, 188,
        # 130, 130, 188,
    ])

    # image = image.quantize(16, 0)
    # image.putpalette([
    #     0, 0, 0,  # black background
    #     # 137, 137, 188,
    #
    #     255, 255, 255,
    #     255, 255, 255,
    #     # 255, 255, 255,
    #     # 255, 255, 255,
    #
    #     # 255, 0, 0,  # index 1 is red
    #     # 255, 255, 0,  # index 2 is yellow
    #     # 255, 153, 0,  # index 3 is orange
    #     # 136, 136, 187,
    #     # 137, 136, 179,
    #     # 137, 137, 188,
    #     # 130, 130, 188,
    # ])

    # print(image.getpalette())
    # image = image.convert('1')

    #
    # (width, height) = (image.width // 2, image.height // 2)
    # image = image.resize((width, height), Image.NEAREST, None, 1)
    #
    # thresh = 120
    # fn = lambda x: 255 if x > thresh else 0
    # image = image.convert('L').point(fn, mode='1')
    #
    # (width, height) = (image.width // 2, image.height // 2)
    # # image = image.resize((width, height), Image.NEAREST, None, 6)
    #
    # image = image.resize((300, 300))

    # image = image.quantize(4, 0)

    im.save('images/2_filtered.gif', 'gif')

    # ndarray = numpy.array(image)
    # # conved = conv(ndarray)
    # conved = conv(ndarray)
    # cim = Image.fromarray(conved)
    # cim = cim.rotate(90)
    #
    # ndarray = numpy.array(cim)
    # # ndarray.transpose()
    # conved = conv(ndarray)
    # cim = Image.fromarray(conved)
    # cim = cim.rotate(-90)
    #
    # cim = cim.resize((width, height), Image.NEAREST, None, 6)
    # # cim = cim.resize((300, 300))
    #
    # cim.save('images/0_conv.gif', 'gif')

    # ndarray = numpy.array(cim)
    # ndarray = ndarray.astype(int)
    # numpy.savetxt("array.txt", ndarray, fmt="%s")


if __name__ == '__main__':
    t = time.perf_counter()

    im: Image = Image.open("../images/22_orig.gif")
    matrix = transform(im, save_debug_images=True, debug_image_name='0')

    elapsed_time = time.perf_counter() - t
    print('elapsed_time total', '%.6f' % elapsed_time)
    # transform2()