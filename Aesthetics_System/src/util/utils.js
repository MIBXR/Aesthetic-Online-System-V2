function rgbToHsl(r, g, b) {
	r /= 255, g /= 255, b /= 255;
	var max = Math.max(r, g, b),
		min = Math.min(r, g, b);
	var h, s, l = (max + min) / 2;

	if (max == min) {
		h = s = 0; // achromatic
	} else {
		var d = max - min;
		s = l > 0.5 ? d / (2 - max - min) : d / (max + min);
		switch (max) {
			case r:
				h = (g - b) / d + (g < b ? 6 : 0);
				break;
			case g:
				h = (b - r) / d + 2;
				break;
			case b:
				h = (r - g) / d + 4;
				break;
		}
		h *= 60;
	}

	return [h, s * 100, l * 100];
}

function hslToRgb(h, s, l) {
	h = h / 360;
	s = s / 100;
	l = l / 100;
	var r, g, b;

	if (s == 0) {
		r = g = b = l; // achromatic
	} else {
		var hue2rgb = function hue2rgb(p, q, t) {
			if (t < 0) t += 1;
			if (t > 1) t -= 1;
			if (t < 1 / 6) return p + (q - p) * 6 * t;
			if (t < 1 / 2) return q;
			if (t < 2 / 3) return p + (q - p) * (2 / 3 - t) * 6;
			return p;
		}

		var q = l < 0.5 ? l * (1 + s) : l + s - l * s;
		var p = 2 * l - q;
		r = hue2rgb(p, q, h + 1 / 3);
		g = hue2rgb(p, q, h);
		b = hue2rgb(p, q, h - 1 / 3);
	}
	return [Math.round(r * 255), Math.round(g * 255), Math.round(b * 255)];
}

function rgbToHex(rgb){
	var color = '#';
	var tmp;
	for(var i=0;i<rgb.length;i++){
		tmp = rgb[i].toString(16);
      		color += tmp.length<2?('0'+tmp):tmp;
	}
	return color;
}

function minpx(x1,x2) {
		console.log('x1', x1)
		if(x1>x2) return Math.abs(x2-20)+'px'
		else return Math.abs(x1-20)+'px'
	}

function leftminpx(x1,x2) {
		if(x1>x2) return -Math.abs((x2-20)/2)+'px'
		else return -Math.abs((x1-20)/2)+'px'
	}

export {
	rgbToHsl,
	hslToRgb,
	rgbToHex,
	minpx,
	leftminpx,
}