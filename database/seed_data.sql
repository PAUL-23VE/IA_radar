-- ============================================================
--  DATOS FICTICIOS - 50 propietarios y vehículos
--  Ejecutar: psql -U postgres -d placas_db -f database/seed_data.sql
--  Placas formato Ecuador: 3 letras + guion + 4 números (ej: ABC-1234)
-- ============================================================

\connect placas_db;

-- ============================================================
--  PROPIETARIOS (ficticios)
-- ============================================================
INSERT INTO propietarios (cedula, nombres, apellidos, email, telefono, direccion) VALUES
('1701234560', 'Carlos Andrés',    'Morales Vega',      'c.morales@gmail.com',      '0987654321', 'Av. Amazonas N35-17, Quito'),
('1702345671', 'María José',       'Torres Salazar',    'm.torres@hotmail.com',     '0991234567', 'Calle Rocafuerte 4-52, Ambato'),
('1703456782', 'Luis Fernando',    'Chávez Ramos',      'l.chavez@outlook.com',     '0976543210', 'Av. Colón 2-14, Quito'),
('1704567893', 'Ana Gabriela',     'Paredes Jiménez',   'ana.paredes@yahoo.com',    '0985432109', 'Calle Bolívar 3-21, Riobamba'),
('1705678904', 'Jorge Eduardo',    'Espinoza Lara',     'j.espinoza@gmail.com',     '0974321098', 'Av. 6 de Diciembre N22-89, Quito'),
('1706789015', 'Patricia Elena',   'Guzmán Flores',     'p.guzman@gmail.com',       '0963210987', 'Calle Sucre 5-67, Latacunga'),
('1707890126', 'Roberto Carlos',   'Mendoza Ortiz',     'r.mendoza@hotmail.com',    '0952109876', 'Av. Real Audiencia 0-31, Quito'),
('1708901237', 'Verónica Paola',   'Castro Herrera',    'v.castro@outlook.com',     '0941098765', 'Calle García Moreno 1-45, Cuenca'),
('1709012348', 'Diego Alejandro',  'Ríos Benítez',      'd.rios@gmail.com',         '0930987654', 'Av. Universitaria 2-78, Quito'),
('1710123459', 'Claudia Beatriz',  'Almeida Paz',       'c.almeida@yahoo.com',      '0929876543', 'Calle 10 de Agosto 6-34, Guaranda'),
('1711234560', 'Miguel Ángel',     'Peñafiel Rojas',    'm.penafiel@gmail.com',     '0918765432', 'Av. Mariscal Sucre N-56, Quito'),
('1712345671', 'Sofía Valentina',  'Naranjo Cabrera',   's.naranjo@hotmail.com',    '0907654321', 'Calle Espejo 7-12, Ibarra'),
('1713456782', 'Andrés Sebastián', 'Villacís Mena',     'a.villacis@outlook.com',   '0896543210', 'Av. Pedro Vicente Maldonado, Quito'),
('1714567893', 'Laura Cristina',   'Bonilla Suárez',    'l.bonilla@gmail.com',      '0885432109', 'Calle Olmedo 3-89, Loja'),
('1715678904', 'Fernando José',    'Aguirre Pilco',     'f.aguirre@yahoo.com',      '0874321098', 'Av. Eloy Alfaro N44-123, Quito'),
('1716789015', 'Mónica Alexandra', 'Cárdenas Luna',     'm.cardenas@gmail.com',     '0863210987', 'Calle Chile 2-56, Esmeraldas'),
('1717890126', 'Pablo Rodrigo',    'Zárate Freire',     'p.zarate@hotmail.com',     '0852109876', 'Av. De los Shyris 3400, Quito'),
('1718901237', 'Isabel Fernanda',  'Montesdeoca Gil',   'i.montesdeoca@gmail.com',  '0841098765', 'Calle Chimborazo 4-67, Machala'),
('1719012348', 'Esteban Mauricio', 'Reinoso Valdez',    'e.reinoso@outlook.com',    '0830987654', 'Av. Patria E4-55, Quito'),
('1720123459', 'Gabriela Mishel',  'Tipán Veloz',       'g.tipan@gmail.com',        '0829876543', 'Calle Vargas Torres 5-78, Santo Domingo'),
('1721234560', 'Ricardo Xavier',   'Sailema Toapanta',  'r.sailema@gmail.com',      '0818765432', 'Av. Simón Bolívar, Quito'),
('1722345671', 'Daniela Estefanía','Núñez Pozo',        'd.nunez@hotmail.com',      '0807654321', 'Calle Padre Solano 1-23, Portoviejo'),
('1723456782', 'Jhonatan David',   'Quishpe Iza',       'j.quishpe@gmail.com',      '0796543210', 'Calle Mejía 6-45, Latacunga'),
('1724567893', 'Ximena Carolina',  'Morejón Sánchez',   'x.morejon@yahoo.com',      '0785432109', 'Av. González Suárez N19-51, Quito'),
('1725678904', 'Héctor Patricio',  'Llerena Orozco',    'h.llerena@outlook.com',    '0774321098', 'Calle Pichincha 3-21, Riobamba');

-- ============================================================
--  VEHÍCULOS  (50 vehículos, placas formato Ecuador)
-- ============================================================
INSERT INTO vehiculos (placa, marca, modelo, anio, color, tipo, cilindraje, propietario_id) VALUES
('ABC-1234', 'Toyota',      'Corolla',        2020, 'Blanco',       'Sedan',    '1800cc', 1),
('BCD-2345', 'Chevrolet',   'Sail',           2019, 'Rojo',         'Sedan',    '1400cc', 2),
('CDE-3456', 'Kia',         'Sportage',       2021, 'Gris Plata',   'SUV',      '2000cc', 3),
('DEF-4567', 'Hyundai',     'Tucson',         2022, 'Negro',        'SUV',      '2400cc', 4),
('EFG-5678', 'Mazda',       'CX-5',           2020, 'Azul',         'SUV',      '2500cc', 5),
('FGH-6789', 'Toyota',      'Hilux',          2021, 'Blanco',       'Camioneta','2700cc', 6),
('GHI-7890', 'Chevrolet',   'D-MAX',          2019, 'Rojo',         'Camioneta','3000cc', 7),
('HIJ-8901', 'Ford',        'F-150',          2022, 'Negro',        'Camioneta','3500cc', 8),
('IJK-9012', 'Nissan',      'Frontier',       2020, 'Gris',         'Camioneta','2500cc', 9),
('JKL-0123', 'Mitsubishi',  'L200',           2021, 'Blanco',       'Camioneta','2400cc', 10),
('KLM-1234', 'Volkswagen',  'Golf',           2019, 'Azul Oscuro',  'Hatchback','1600cc', 11),
('LMN-2345', 'Honda',       'Civic',          2022, 'Plata',        'Sedan',    '1500cc', 12),
('MNO-3456', 'Suzuki',      'Grand Vitara',   2020, 'Verde',        'SUV',      '2400cc', 13),
('NOP-4567', 'Renault',     'Duster',         2021, 'Naranja',      'SUV',      '1600cc', 14),
('OPQ-5678', 'Kia',         'Picanto',        2019, 'Rojo',         'Hatchback','1200cc', 15),
('PQR-6789', 'Toyota',      'RAV4',           2022, 'Blanco',       'SUV',      '2500cc', 16),
('QRS-7890', 'Chevrolet',   'Tracker',        2021, 'Gris',         'SUV',      '1000cc', 17),
('RST-8901', 'Hyundai',     'Santa Fe',       2020, 'Negro',        'SUV',      '2400cc', 18),
('STU-9012', 'Ford',        'Escape',         2019, 'Plata',        'SUV',      '1500cc', 19),
('TUV-0123', 'Mazda',       'Mazda3',         2022, 'Azul',         'Sedan',    '2000cc', 20),
('UVW-1234', 'Nissan',      'Sentra',         2020, 'Blanco',       'Sedan',    '1800cc', 21),
('VWX-2345', 'Honda',       'HR-V',           2021, 'Gris Plata',   'SUV',      '1800cc', 22),
('WXY-3456', 'Suzuki',      'Swift',          2019, 'Amarillo',     'Hatchback','1200cc', 23),
('XYZ-4567', 'Volkswagen',  'Tiguan',         2022, 'Blanco',       'SUV',      '2000cc', 24),
('YZA-5678', 'Kia',         'Sorento',        2020, 'Negro',        'SUV',      '3500cc', 25),
('ZAB-6789', 'Toyota',      'Fortuner',       2021, 'Blanco',       'SUV',      '2700cc', 1),
('ABD-7890', 'Chevrolet',   'Captiva',        2019, 'Rojo',         'SUV',      '2400cc', 2),
('ACE-8901', 'Mitsubishi',  'Outlander',      2022, 'Plata',        'SUV',      '2000cc', 3),
('ADF-9012', 'Renault',     'Logan',          2020, 'Beige',        'Sedan',    '1400cc', 4),
('ADG-0123', 'Ford',        'EcoSport',       2021, 'Azul',         'SUV',      '1500cc', 5),
('AEH-1234', 'Toyota',      'Yaris',          2019, 'Rojo',         'Hatchback','1500cc', 6),
('AFI-2345', 'Hyundai',     'Elantra',        2022, 'Gris',         'Sedan',    '2000cc', 7),
('AGJ-3456', 'Kia',         'Stinger',        2021, 'Negro',        'Sedan',    '3300cc', 8),
('AHK-4567', 'Mazda',       'CX-9',           2020, 'Blanco',       'SUV',      '2500cc', 9),
('AIL-5678', 'Chevrolet',   'Onix',           2022, 'Plata',        'Hatchback','1000cc', 10),
('AJM-6789', 'Nissan',      'X-Trail',        2019, 'Negro',        'SUV',      '2000cc', 11),
('AKN-7890', 'Honda',       'CR-V',           2021, 'Blanco',       'SUV',      '1500cc', 12),
('ALO-8901', 'Suzuki',      'Vitara',         2020, 'Gris',         'SUV',      '1600cc', 13),
('AMP-9012', 'Toyota',      'Land Cruiser',   2022, 'Blanco',       'SUV',      '4000cc', 14),
('ANQ-0123', 'Ford',        'Territory',      2021, 'Rojo',         'SUV',      '1500cc', 15),
('AOR-1234', 'Hyundai',     'Ioniq',          2022, 'Azul',         'Sedan',    'Eléctrico', 16),
('APS-2345', 'Kia',         'EV6',            2022, 'Gris Plata',   'SUV',      'Eléctrico', 17),
('AQT-3456', 'Volkswagen',  'Jetta',          2020, 'Negro',        'Sedan',    '1400cc', 18),
('ARU-4567', 'Mazda',       'BT-50',          2021, 'Blanco',       'Camioneta','3200cc', 19),
('ASV-5678', 'Toyota',      'Avanza',         2019, 'Plata',        'Minivan',  '1500cc', 20),
('ATW-6789', 'Chevrolet',   'Spin',           2020, 'Blanco',       'Minivan',  '1800cc', 21),
('AUX-7890', 'Nissan',      'NV200',          2021, 'Blanco',       'Minivan',  '1600cc', 22),
('AVY-8901', 'Renault',     'Kangoo',         2019, 'Gris',         'Minivan',  '1600cc', 23),
('AWZ-9012', 'Kia',         'Carnival',       2022, 'Negro',        'Minivan',  '2200cc', 24),
('AXA-0123', 'Toyota',      'Hiace',          2021, 'Blanco',       'Van',      '2700cc', 25);
