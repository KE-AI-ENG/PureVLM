# setup.py

from setuptools import setup, find_packages

setup(
    # 基本信息
    name='purevlm',
    version='0.1.0',
    author='MLSys',
    description='PureVLM - A Vision Language Model inference engine based on Qwen3VL',
    
    # 项目分类
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    
    # 关键词
    keywords='vision-language-model, vlm, qwen3vl, multimodal, deep-learning, pytorch',
    
    # Python 版本要求
    python_requires='>=3.8',
    
    # 包发现
    packages=find_packages(exclude=['tests', 'tests.*', 'examples', 'examples.*']),
    
    # 包数据
    package_data={
        'purevlm': [
            'configs/*.json',
            'configs/*.yaml',
        ],
    },
    
    # 命令行入口点
    entry_points={
        'console_scripts': [
            'purevlm-serve=purevlm.serve:main',
        ],
    },

    # 许可证
    license='MIT License',
)